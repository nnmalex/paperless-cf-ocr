#!/usr/bin/env python3
"""Milestone-1 lab: probe Cloudflare AI Gateway endpoints and compare OCR models.

Subcommands
-----------
probe   Send one small image to every endpoint/header variant and report which work.
eval    Run each model over a samples directory, save transcriptions, and score them
        against hand-checked ground truth (<name>.gt.txt or <name>.p<N>.gt.txt).

Configuration (env vars, or a .env file next to this script):
    CF_ACCOUNT_ID       Cloudflare account id
    CF_GATEWAY_ID       AI Gateway id (slug)
    CF_API_TOKEN        API token with "Workers AI - Read" (sent as Authorization)
    CF_AIG_TOKEN        optional, for authenticated gateways (sent as cf-aig-authorization)

Only the standard library is required. Optional: Pillow (resizing), pypdfium2 (PDF pages).
"""
from __future__ import annotations

import argparse
import base64
import concurrent.futures as cf
import csv
import io
import json
import mimetypes
import os
import re
import sys
import time
import unicodedata
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
GATEWAY_HOST = "https://gateway.ai.cloudflare.com/v1"

PROMPT = (
    "Transcribe all text in this document image exactly as written, in natural reading "
    "order. Preserve line breaks and paragraphs. Render tables as rows with cells "
    "separated by ' | '. Do not translate, summarize, correct, or add commentary. "
    "If there is no text, output nothing."
)

# name -> (model id, api style). "native" = Workers AI model schema; "openai" = chat vision.
MODELS: dict[str, tuple[str, str]] = {
    "moondream": ("@cf/moondream/moondream3.1-9B-A2B", "native-moondream"),
    "mistral-small": ("@cf/mistralai/mistral-small-3.1-24b-instruct", "openai"),
    "llama4-scout": ("@cf/meta/llama-4-scout-17b-16e-instruct", "openai"),
    "qwen3.8-27b": ("@cf/qwen/qwen3.8-27b", "openai"),
    "gemma4-26b": ("@cf/google/gemma-4-26b-a4b-it", "openai"),
    # Third-party models via the gateway's unified OpenAI-compatible route (billed through
    # Cloudflare). Sonnet 5.5 rejects non-default temperature, so none is sent.
    "sonnet-5.5": ("anthropic/claude-sonnet-5-5", "compat-3p"),
}
DEFAULT_EVAL = ["moondream", "mistral-small", "llama4-scout", "qwen3.8-27b"]


# --------------------------------------------------------------------------- config


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip("'\""))


@dataclass
class GW:
    account: str
    gateway: str
    token: str
    aig_token: str | None

    @classmethod
    def from_env(cls) -> "GW":
        load_dotenv(HERE / ".env")
        missing = [k for k in ("CF_ACCOUNT_ID", "CF_GATEWAY_ID", "CF_API_TOKEN") if not os.environ.get(k)]
        if missing:
            sys.exit(f"Missing env vars: {', '.join(missing)} (see tools/lab/.env.example)")
        return cls(
            os.environ["CF_ACCOUNT_ID"],
            os.environ["CF_GATEWAY_ID"],
            os.environ["CF_API_TOKEN"],
            os.environ.get("CF_AIG_TOKEN") or None,
        )

    @property
    def base(self) -> str:
        return f"{GATEWAY_HOST}/{self.account}/{self.gateway}"

    def headers(self, aig: bool = True) -> dict[str, str]:
        # Cloudflare's bot protection rejects urllib's default UA with "error code: 1010".
        h = {"Authorization": f"Bearer {self.token}", "Content-Type": "application/json",
             "User-Agent": "paperless-cf-ocr-lab/0.1",
             # Repeated identical probe/eval requests must not be served from gateway cache.
             "cf-aig-skip-cache": "true"}
        if aig and self.aig_token:
            h["cf-aig-authorization"] = f"Bearer {self.aig_token}"
        return h


# --------------------------------------------------------------------------- http


@dataclass
class Resp:
    status: int
    body: Any
    seconds: float
    headers: dict[str, str] = field(default_factory=dict)


def post(url: str, headers: dict[str, str], payload: dict, timeout: float = 180) -> Resp:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw, status, hdrs = r.read(), r.status, dict(r.headers)
    except urllib.error.HTTPError as e:
        raw, status, hdrs = e.read(), e.code, dict(e.headers or {})
    except (urllib.error.URLError, TimeoutError) as e:
        return Resp(0, {"error": str(e)}, time.monotonic() - t0)
    dt = time.monotonic() - t0
    try:
        body = json.loads(raw)
    except ValueError:
        body = raw.decode(errors="replace")
    return Resp(status, body, dt, hdrs)


# --------------------------------------------------------------------------- images


def to_jpeg_data_uri(img_bytes: bytes, mime: str, max_px: int) -> tuple[str, int]:
    """Downscale to max_px on the long edge if Pillow is available; return (data URI, bytes)."""
    try:
        from PIL import Image  # type: ignore

        im = Image.open(io.BytesIO(img_bytes))
        im = im.convert("L") if im.mode in ("1", "L", "LA") else im.convert("RGB")
        if max(im.size) > max_px:
            im.thumbnail((max_px, max_px))
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=88)
        img_bytes, mime = buf.getvalue(), "image/jpeg"
    except ImportError:
        pass
    return f"data:{mime};base64,{base64.b64encode(img_bytes).decode()}", len(img_bytes)


def document_pages(path: Path, dpi: int) -> list[tuple[bytes, str]]:
    """Return [(image bytes, mime)] for an image or each page of a PDF."""
    if path.suffix.lower() == ".pdf":
        try:
            import pypdfium2 as pdfium  # type: ignore
        except ImportError:
            sys.exit("PDF input needs pypdfium2: pip install -r tools/lab/requirements.txt")
        pages = []
        pdf = pdfium.PdfDocument(str(path))
        for page in pdf:
            pil = page.render(scale=dpi / 72).to_pil()
            buf = io.BytesIO()
            pil.save(buf, format="PNG")
            pages.append((buf.getvalue(), "image/png"))
        return pages
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    return [(path.read_bytes(), mime)]


def tiny_test_image() -> str:
    """A small image with real text, for probing. Uses Pillow if present, else a 1x1 PNG."""
    try:
        from PIL import Image, ImageDraw  # type: ignore

        im = Image.new("RGB", (480, 120), "white")
        d = ImageDraw.Draw(im)
        d.text((20, 40), "INVOICE 2026-0042  Total: 123.45 EUR", fill="black")
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        raw = buf.getvalue()
    except ImportError:
        raw = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
        )
    return "data:image/png;base64," + base64.b64encode(raw).decode()


# --------------------------------------------------------------------------- model calls


def build_request(gw: GW, model: str, style: str, image_uri: str, max_tokens: int,
                  variant: str = "default", prompt: str | None = None) -> tuple[str, dict]:
    if style == "native-moondream":
        url = f"{gw.base}/workers-ai/{model}"
        payload = {
            "task": "query",
            "image": image_uri,
            "question": prompt or PROMPT,
            "reasoning": False,
            "temperature": 0,
            "max_tokens": max_tokens,
            # Non-streaming calls currently return an empty `result: {}` (2026-11-20);
            # streaming works, so always stream and reassemble (see parse_moondream_sse).
            "stream": True,
        }
        if variant == "direct":
            url = f"https://api.cloudflare.com/client/v4/accounts/{gw.account}/ai/run/{model}"
        return url, payload
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": prompt or PROMPT},
            {"type": "image_url", "image_url": {"url": image_uri}},
        ],
    }]
    if style == "compat-3p":
        return f"{gw.base}/compat/chat/completions", {
            "model": model, "messages": messages, "max_tokens": max_tokens,
        }
    if variant == "direct":
        return f"https://api.cloudflare.com/client/v4/accounts/{gw.account}/ai/v1/chat/completions", {
            "model": model, "messages": messages, "temperature": 0, "max_tokens": max_tokens,
        }
    if variant == "compat":
        return f"{gw.base}/compat/chat/completions", {
            "model": f"workers-ai/{model}", "messages": messages,
            "temperature": 0, "max_tokens": max_tokens,
        }
    return f"{gw.base}/workers-ai/v1/chat/completions", {
        "model": model, "messages": messages, "temperature": 0, "max_tokens": max_tokens,
    }


def parse_moondream_sse(raw: str) -> dict | None:
    """Moondream streams cumulative `chunk.answer` values; the final chunk has finish_reason."""
    last, usage = None, {}
    for line in raw.splitlines():
        if not line.startswith("data:") or line.strip() == "data: [DONE]":
            continue
        try:
            ev = json.loads(line[5:])
        except ValueError:
            continue
        if isinstance(ev.get("chunk"), dict):
            last = ev["chunk"]
        if ev.get("usage", {}).get("completion_tokens"):
            usage = ev["usage"]
    if last is None:
        return None
    return {"result": {"answer": last.get("answer") or "", "finish_reason": last.get("finish_reason")},
            "usage": usage}


def extract_text(body: Any) -> str | None:
    if isinstance(body, str) and "data:" in body:
        body = parse_moondream_sse(body)
    if not isinstance(body, dict):
        return None
    if "choices" in body:  # OpenAI style
        try:
            return body["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError):
            return None
    res = body.get("result", body)
    if isinstance(res, dict):
        for k in ("answer", "response", "caption", "text"):
            if isinstance(res.get(k), str):
                return res[k]
    return None


def extract_usage(body: Any) -> dict:
    if isinstance(body, str) and "data:" in body:
        body = parse_moondream_sse(body)
    if not isinstance(body, dict):
        return {}
    u = body.get("usage") or (body.get("result") or {}).get("usage") or {}
    m = (body.get("result") or {}).get("metrics") if isinstance(body.get("result"), dict) else None
    out = {}
    for src in (u, m or {}):
        for k in ("prompt_tokens", "completion_tokens", "input_tokens", "output_tokens", "total_tokens"):
            if k in src:
                out[k] = src[k]
    return out


def err_summary(r: Resp) -> str:
    b = r.body
    if isinstance(b, dict):
        if "errors" in b and b["errors"]:
            return json.dumps(b["errors"])[:300]
        if "error" in b:
            return json.dumps(b["error"])[:300]
    return str(b)[:300].replace("\n", " ")


# --------------------------------------------------------------------------- probe


def cmd_probe(args: argparse.Namespace) -> None:
    gw = GW.from_env()
    img = tiny_test_image()
    names = args.models or list(MODELS)
    print(f"Gateway: {gw.base}  (cf-aig-authorization: {'yes' if gw.aig_token else 'no'})\n")
    rows = []
    for name in names:
        model, style = MODELS[name]
        variants = ["default", "direct"] if style.startswith("native") else ["default", "compat", "direct"]
        for v in variants:
            url, payload = build_request(gw, model, style, img, 256, v)
            r = post(url, gw.headers(), payload, timeout=90)
            text = extract_text(r.body) if r.status == 200 else None
            ok = r.status == 200 and text is not None
            cache = r.headers.get("cf-aig-cache-status", "")
            rows.append((name, v, url.replace(gw.base, "{gw}"), r.status, f"{r.seconds:.1f}s",
                         cache, (text or err_summary(r)).replace("\n", " ")[:90]))
            print(f"[{'OK ' if ok else 'ERR'}] {name:14} {v:8} {r.status:>3} {r.seconds:5.1f}s  "
                  f"{(text or err_summary(r)).replace(chr(10), ' ')[:90]}")
    # Header check: if a gateway token is configured, confirm the gateway actually enforces it.
    if gw.aig_token:
        model, style = MODELS["mistral-small"]
        url, payload = build_request(gw, model, style, img, 64)
        h = {k: v for k, v in gw.headers().items() if k != "Authorization"}
        r = post(url, h, payload, timeout=60)
        print(f"\ncf-aig-authorization only (no Authorization) -> HTTP {r.status}")
        h = {k: v for k, v in gw.headers(aig=False).items() if k != "Authorization"}
        r = post(url, h, payload, timeout=60)
        print(f"no credentials at all -> HTTP {r.status} "
              f"({'gateway is authenticated' if r.status in (401, 403) else 'gateway did NOT reject'})")
    out = HERE / "results" / "probe.csv"
    out.parent.mkdir(exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "variant", "url", "status", "latency", "aig_cache", "text_or_error"])
        w.writerows(rows)
    print(f"\nSaved {out}")


# --------------------------------------------------------------------------- scoring


def normalize(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    s = s.translate(str.maketrans({"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
                                   "\u2013": "-", "\u2014": "-", "\u2022": " ", "\u25cf": " "}))
    s = re.sub(r"^\s*[-•*]\s+", " ", s, flags=re.M)  # bullets
    s = re.sub(r"[*_`#>|]+", " ", s)  # markdown/table noise shouldn't count as errors
    return re.sub(r"\s+", " ", s).strip().lower()


def levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(hyp: str, ref: str) -> float:
    h, r = normalize(hyp), normalize(ref)
    return levenshtein(h, r) / max(1, len(r))


def wer(hyp: str, ref: str) -> float:
    h, r = normalize(hyp).split(), normalize(ref).split()
    # word-level levenshtein via mapping words to chars
    vocab: dict[str, str] = {}
    enc = lambda ws: "".join(vocab.setdefault(w, chr(0xE000 + len(vocab))) for w in ws)  # noqa: E731
    return levenshtein(enc(h), enc(r)) / max(1, len(r))


def word_f1(hyp: str, ref: str) -> float:
    """Order-insensitive word overlap (multiset F1): robust to column/reading-order choices."""
    from collections import Counter
    h, r = Counter(normalize(hyp).split()), Counter(normalize(ref).split())
    tp = sum((h & r).values())
    if not h or not r:
        return 0.0
    prec, rec = tp / sum(h.values()), tp / sum(r.values())
    return 0.0 if tp == 0 else 2 * prec * rec / (prec + rec)


def find_gt(doc: Path, page_idx: int, n_pages: int) -> str | None:
    cands = [doc.with_name(f"{doc.stem}.p{page_idx + 1}.gt.txt")]
    if n_pages == 1:
        cands.insert(0, doc.with_name(f"{doc.stem}.gt.txt"))
    for c in cands:
        if c.exists():
            return c.read_text()
    return None


# --------------------------------------------------------------------------- eval


def cmd_eval(args: argparse.Namespace) -> None:
    gw = GW.from_env()
    samples = Path(args.samples)
    docs = sorted(p for p in samples.glob(args.include)
                  if p.suffix.lower() in {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp"})
    if not docs:
        sys.exit(f"No documents in {samples}")
    out_dir = HERE / "results" / time.strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True)

    jobs = []  # (doc, page_idx, n_pages, data_uri, bytes)
    for d in docs:
        pages = document_pages(d, args.dpi)[: args.max_pages]
        for i, (raw, mime) in enumerate(pages):
            uri, n = to_jpeg_data_uri(raw, mime, args.max_px)
            jobs.append((d, i, len(pages), uri, n))
    print(f"{len(docs)} documents, {len(jobs)} pages, models: {', '.join(args.models)}")

    def run(name: str, job: tuple) -> dict:
        d, i, n_pages, uri, nbytes = job
        model, style = MODELS[name]
        url, payload = build_request(gw, model, style, uri, args.max_tokens, args.variant,
                                     Path(args.prompt_file).read_text() if args.prompt_file else None)
        payload.update(json.loads(args.extra_params) if args.extra_params else {})
        for attempt in range(4):
            r = post(url, gw.headers(), payload)
            if r.status not in (408, 429, 500, 502, 503, 504, 0):
                break
            time.sleep(2 ** attempt)
        text = extract_text(r.body) if r.status == 200 else None
        rec = {"model": name, "doc": d.name, "page": i + 1, "status": r.status,
               "seconds": round(r.seconds, 2), "image_kb": nbytes // 1024,
               "chars": len(text or ""), **extract_usage(r.body)}
        if text is None:
            rec["error"] = err_summary(r)
        else:
            (out_dir / name).mkdir(exist_ok=True)
            (out_dir / name / f"{d.stem}.p{i + 1}.txt").write_text(text)
            gt = find_gt(d, i, n_pages)
            if gt is not None:
                rec["cer"] = round(cer(text, gt), 4)
                rec["wer"] = round(wer(text, gt), 4)
                rec["word_f1"] = round(word_f1(text, gt), 4)
        return rec

    records = []
    with cf.ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = [ex.submit(run, m, j) for m in args.models for j in jobs]
        for k, f in enumerate(cf.as_completed(futs), 1):
            rec = f.result()
            records.append(rec)
            tag = f"cer={rec['cer']:.3f}" if "cer" in rec else rec.get("error", "")[:60]
            print(f"  [{k}/{len(futs)}] {rec['model']:14} {rec['doc']} p{rec['page']} "
                  f"{rec['status']} {rec['seconds']}s {tag}")

    keys = sorted({k for r in records for k in r})
    with (out_dir / "pages.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(sorted(records, key=lambda r: (r["model"], r["doc"], r["page"])))

    lines = ["| model | pages ok | errors | mean CER | mean WER | word F1 | p50 latency | p95 latency | out tokens/page |",
             "|---|---|---|---|---|---|---|---|---|"]
    for name in args.models:
        rs = [r for r in records if r["model"] == name]
        ok = [r for r in rs if "error" not in r]
        lat = sorted(r["seconds"] for r in ok)
        pct = lambda p: f"{lat[min(len(lat) - 1, int(p * len(lat)))]:.1f}s" if lat else "-"  # noqa: E731
        mean = lambda k: (f"{sum(r[k] for r in ok if k in r) / n:.3f}"  # noqa: E731
                          if (n := sum(1 for r in ok if k in r)) else "-")
        toks = [r.get("completion_tokens") or r.get("output_tokens") for r in ok]
        toks = [t for t in toks if t]
        lines.append(f"| {name} | {len(ok)} | {len(rs) - len(ok)} | {mean('cer')} | {mean('wer')} | {mean('word_f1')} | "
                     f"{pct(0.5)} | {pct(0.95)} | {round(sum(toks) / len(toks)) if toks else '-'} |")
    summary = "\n".join(lines)
    (out_dir / "summary.md").write_text(summary + "\n")
    print("\n" + summary + f"\n\nResults in {out_dir}")


# --------------------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("probe", help="check endpoint/header variants with a tiny image")
    p.add_argument("--models", nargs="*", choices=list(MODELS))
    p.set_defaults(fn=cmd_probe)

    e = sub.add_parser("eval", help="compare models on a samples directory")
    e.add_argument("samples", help="directory with PDFs/images (+ optional *.gt.txt)")
    e.add_argument("--models", nargs="*", default=DEFAULT_EVAL, choices=list(MODELS))
    e.add_argument("--variant", choices=["default", "compat", "direct"], default="default",
                   help="default/compat = through AI Gateway; direct = Workers AI REST, bypassing it")
    e.add_argument("--include", default="*", help="glob within samples dir, e.g. 'lv*'")
    e.add_argument("--prompt-file", help="override the transcription prompt")
    e.add_argument("--extra-params", help='JSON merged into each request, e.g. \'{"reasoning_effort": "low"}\'')
    e.add_argument("--dpi", type=int, default=200)
    e.add_argument("--max-px", type=int, default=2048)
    e.add_argument("--max-pages", type=int, default=5, help="per document")
    e.add_argument("--max-tokens", type=int, default=4096)
    e.add_argument("--concurrency", type=int, default=4)
    e.set_defaults(fn=cmd_eval)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
