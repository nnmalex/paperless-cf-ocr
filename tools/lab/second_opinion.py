#!/usr/bin/env python3
"""Milestone-1b lab: does a confidence-triggered second opinion improve OCR?

For every page:
  1. Primary model (strict prompt, logprobs) -> text + per-word confidence.
  2. If the page is low-confidence, run a second model, then an adjudication call
     (image + both transcriptions + the primary's uncertain words) on the primary model.
Writes one results dir with variants usable by score_fields.py:
  primary/, final-<second>/ (adjudicated for flagged pages, primary text otherwise),
  second-<second>/ (raw second-model output, flagged pages only), plus pages.json.

usage: second_opinion.py [--seconds mistral-small llama4-scout] [--all]   (--all = escalate every page)
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import math
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cfocr_lab as L  # noqa: E402

STRICT = (HERE / "prompt_strict.txt").read_text()

ADJUDICATE = """\
You are given a document page image and two independent OCR transcriptions of it, A and B.
Produce the single most accurate transcription of the page.

Rules:
- The image is the only source of truth. Where A and B differ, look at the image and keep the reading that matches it.
- Start from A's layout and order; take words or lines from B only where the image shows B is right or A missed visible text.
- Never add text that is not visible in the image, and never fill in text from what similar documents usually say.
- If the image does not let you decide between the readings, or neither matches, write [illegible].
- Ignore faint mirror-image text that shows through from the other side of the paper.
- Output only the final transcription, no commentary.

Words that A's OCR engine was least sure about (check these against the image): {uncertain}

=== Transcription A ===
{a}

=== Transcription B ===
{b}
"""


def word_confidences(logprobs: list[dict]) -> list[tuple[str, float]]:
    """Group tokens into whitespace-delimited words (min token prob); drop reasoning."""
    words: list[tuple[str, float]] = []
    cur, cp = "", 1.0
    for t in logprobs:
        s, p = t["token"], math.exp(t["logprob"])
        if s[:1].isspace():
            if cur.strip():
                words.append((cur.strip(), cp))
            cur, cp = s, p
        else:
            cur, cp = cur + s, min(cp, p)
    if cur.strip():
        words.append((cur.strip(), cp))
    cut = max((i for i, (w, _) in enumerate(words) if "</think>" in w), default=-1)
    return [(w.replace("<|im_end|>", ""), p) for w, p in words[cut + 1:]]


def page_confidence(words: list[tuple[str, float]], text: str) -> dict:
    n = max(1, len(words))
    low = [w for w in words if w[1] < 0.5]
    return {
        "mean": round(sum(p for _, p in words) / n, 4),
        "low_ratio": round(len(low) / n, 4),
        "min": round(min((p for _, p in words), default=1.0), 4),
        "illegible": text.lower().count("[illegible]"),
        "uncertain": [w for w, _ in sorted(words, key=lambda x: x[1])[:20] if _ < 0.6],
    }


def flagged(c: dict) -> bool:
    return c["low_ratio"] > 0.01 or c["min"] < 0.2 or c["illegible"] > 0


def chat(gw: L.GW, model: str, uri: str, prompt: str, logprobs: bool = False,
         max_tokens: int = 4096) -> tuple[str | None, dict, float]:
    url, payload = L.build_request(gw, model, "openai", uri, max_tokens, "default", prompt)
    if logprobs:
        payload["logprobs"] = True
    t0 = time.monotonic()
    for attempt in range(4):
        r = L.post(url, gw.headers(), payload, timeout=240)
        if r.status not in (408, 429, 500, 502, 503, 504, 0):
            break
        time.sleep(2 ** attempt)
    body = r.body if isinstance(r.body, dict) else {}
    if r.status != 200:
        print(f"    ! {model.split('/')[-1]} HTTP {r.status}: {L.err_summary(r)[:120]}")
        return None, body, time.monotonic() - t0
    return L.extract_text(body), body, time.monotonic() - t0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--primary", default="qwen3.8-27b")
    ap.add_argument("--seconds", nargs="*", default=["mistral-small", "llama4-scout"])
    ap.add_argument("--all", action="store_true", help="escalate every page (measure harm)")
    ap.add_argument("--include", default="*.pdf")
    args = ap.parse_args()

    gw = L.GW.from_env()
    pmodel = L.MODELS[args.primary][0]
    out = HERE / "results" / ("so-" + time.strftime("%Y%m%d-%H%M%S"))
    out.mkdir(parents=True)

    pages = []
    for d in sorted((HERE / "samples").glob(args.include)):
        for i, (raw, mime) in enumerate(L.document_pages(d, 200)):
            uri, _ = L.to_jpeg_data_uri(raw, mime, 2048)
            pages.append((d.stem, i + 1, uri))

    def do_page(job: tuple[str, int, str]) -> dict:
        stem, pno, uri = job
        name = f"{stem}.p{pno}"
        rec: dict = {"page": name, "cost_calls": 1}
        text, body, secs = chat(gw, pmodel, uri, STRICT, logprobs=True)
        if text is None:
            rec["error"] = "primary failed"
            return rec
        lp = (body.get("choices") or [{}])[0].get("logprobs") or {}
        conf = page_confidence(word_confidences(lp.get("content") or []), text)
        rec.update(primary_seconds=round(secs, 1), confidence=conf, flagged=flagged(conf))
        (out / "primary").mkdir(exist_ok=True)
        (out / "primary" / f"{name}.txt").write_text(text)
        for sec in args.seconds:
            final = text
            if rec["flagged"] or args.all:
                stext, _, ssecs = chat(gw, L.MODELS[sec][0], uri, STRICT)
                rec[f"{sec}_seconds"] = round(ssecs, 1)
                if stext:
                    (out / f"second-{sec}").mkdir(exist_ok=True)
                    (out / f"second-{sec}" / f"{name}.txt").write_text(stext)
                    prompt = ADJUDICATE.format(uncertain=", ".join(conf["uncertain"]) or "(none)",
                                               a=text, b=stext)
                    atext, _, asecs = chat(gw, pmodel, uri, prompt)
                    rec[f"adjudicate_{sec}_seconds"] = round(asecs, 1)
                    if atext:
                        final = atext
                    else:
                        rec[f"adjudicate_{sec}"] = "failed; kept primary"
                else:
                    rec[f"{sec}"] = "failed; kept primary"
            (out / f"final-{sec}").mkdir(exist_ok=True)
            (out / f"final-{sec}" / f"{name}.txt").write_text(final)
        c = rec["confidence"]
        print(f"  {name:16} flagged={rec['flagged']!s:5} mean={c['mean']:.3f} "
              f"low={c['low_ratio']:.1%} min={c['min']:.2f} illegible={c['illegible']}")
        return rec

    with cf.ThreadPoolExecutor(4) as ex:
        recs = list(ex.map(do_page, pages))
    (out / "pages.json").write_text(json.dumps(recs, indent=2))
    print(f"\nResults in {out}")


if __name__ == "__main__":
    main()
