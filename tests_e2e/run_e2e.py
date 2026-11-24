#!/usr/bin/env python3
"""Drive a running Paperless-ngx (see compose*.yml) through its REST API and check the plugin.

usage: COMPOSE="docker compose -f compose.yml -f compose.installer.yml -f compose.mock.yml" \
       python run_e2e.py            # mock gateway
       E2E_LIVE=1 python run_e2e.py # real gateway: checks transcriptions for real words
Needs: httpx, pillow, fpdf2.
"""

from __future__ import annotations

import os
import re
import shlex
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx

BASE = os.environ.get("PAPERLESS_URL", "http://localhost:8000")
AUTH = ("admin", "e2e-password")
LIVE = os.environ.get("E2E_LIVE") == "1"
MOCK = os.environ.get("MOCK_URL", "http://localhost:8080")
COMPOSE = shlex.split(os.environ.get("COMPOSE", "docker compose"))
HERE = Path(__file__).resolve().parent
failures: list[str] = []


def check(cond: bool, msg: str) -> None:
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        failures.append(msg)


def make_docs(d: Path) -> dict[str, Path]:
    from fpdf import FPDF
    from PIL import Image, ImageDraw, ImageFont

    def page(lines: list[str]) -> Image.Image:
        im = Image.new("L", (1700, 2200), 255)
        draw, font = ImageDraw.Draw(im), ImageFont.load_default(size=64)
        for i, line in enumerate(lines):
            draw.text((150, 200 + i * 130), line, fill=0, font=font)
        return im

    pages = [["SCANNED PAGE ONE", "INVOICE 4711", "TOTAL 123.45 EUR"],
             ["SCANNED PAGE TWO", "THANK YOU FOR", "YOUR BUSINESS"]]
    pdf = FPDF(unit="pt", format="letter")
    for i, lines in enumerate(pages):
        img = d / f"p{i}.png"
        page(lines).save(img, dpi=(200, 200))
        pdf.add_page()
        pdf.image(str(img), x=0, y=0, w=pdf.w, h=pdf.h)
    docs = {"scan.pdf": d / "e2e-scan.pdf", "photo.png": d / "e2e-photo.png",
            "digital.pdf": d / "e2e-digital.pdf"}
    pdf.output(str(docs["scan.pdf"]))
    page(["RECEIPT 98765", "PAID IN FULL"]).save(docs["photo.png"], dpi=(200, 200))
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("helvetica", size=12)
    for i in range(25):
        pdf.cell(0, 8, f"Born digital statement line {i}: 99.99 EUR", new_x="LMARGIN", new_y="NEXT")
    pdf.output(str(docs["digital.pdf"]))
    return docs


def wait_for_api(client: httpx.Client, timeout: float = 900) -> None:
    t0, last = time.monotonic(), ""
    while time.monotonic() - t0 < timeout:
        try:
            r = client.get("/api/documents/")
            state = f"HTTP {r.status_code}"
            if r.status_code == 200:
                print(f"Paperless API ready after {time.monotonic() - t0:.0f}s", flush=True)
                return
        except httpx.HTTPError as e:
            state = type(e).__name__
        if state != last:
            print(f"  waiting for Paperless API: {state} ({time.monotonic() - t0:.0f}s)", flush=True)
            last = state
        time.sleep(5)
    print(logs()[-6000:])
    sys.exit("Paperless did not become ready")


def consume(client: httpx.Client, path: Path, timeout: float = 300) -> dict:
    with path.open("rb") as fh:
        r = client.post("/api/documents/post_document/", files={"document": (path.name, fh)})
    r.raise_for_status()
    task_id = r.json()
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        tasks = client.get("/api/tasks/", params={"task_id": task_id}).json()
        task = (tasks.get("results", tasks) if isinstance(tasks, dict) else tasks)
        task = task[0] if task else {}
        # API v10 (Paperless 3.5 default): lowercase status, related_document_ids list;
        # API v9: uppercase Celery states, related_document.
        status = str(task.get("status", "")).lower()
        if status in ("success", "failure", "revoked"):
            print(f"  {path.name}: {status} after {time.monotonic() - t0:.0f}s", flush=True)
            if status != "success":
                return {"_error": task.get("result") or task.get("result_message") or task}
            ids = task.get("related_document_ids") or [task.get("related_document")]
            doc_id = ids[0]
            return client.get(f"/api/documents/{doc_id}/").json() | {"_id": doc_id}
        time.sleep(3)
    return {"_error": "timeout"}


def logs() -> str:
    return subprocess.run([*COMPOSE, "logs", "--no-color", "webserver"], capture_output=True,
                          text=True, check=False).stdout


def main() -> None:
    client = httpx.Client(base_url=BASE, auth=AUTH, timeout=60)
    wait_for_api(client)
    mock_before = None if LIVE else httpx.get(f"{MOCK}/stats").json()

    with tempfile.TemporaryDirectory() as tmp:
        docs = make_docs(Path(tmp))
        results = {name: consume(client, path) for name, path in docs.items()}

    print("\nDocuments:")
    for name, doc in results.items():
        check("_error" not in doc, f"{name} consumed ({doc.get('_error', '')})")
    scan, photo, digital = results["scan.pdf"], results["photo.png"], results["digital.pdf"]

    if LIVE:
        content = " ".join(scan.get("content", "").upper().split())
        check("INVOICE 4711" in content and "THANK YOU" in content,
              f"scan content transcribed by the model: {content[:120]!r}")
        check("RECEIPT 98765" in photo.get("content", "").upper(), "photo content transcribed")
    else:
        check(scan.get("content", "").count("E2E MOCK OCR") == 2,
              f"scan content comes from the gateway (2 pages): {scan.get('content', '')[:120]!r}")
        check(photo.get("content", "").startswith("E2E MOCK OCR"), "photo content comes from the gateway")
    check("Born digital statement line 24" in digital.get("content", ""),
          "born-digital PDF keeps its embedded text")
    check(bool(scan.get("archived_file_name")), "scan has a hybrid archive PDF")
    check(bool(photo.get("archived_file_name")), "photo has a hybrid archive PDF")
    for name in ("scan.pdf", "photo.png"):
        doc = results[name]
        if "_id" in doc:
            r = client.get(f"/api/documents/{doc['_id']}/thumb/")
            check(r.status_code == 200 and len(r.content) > 100, f"{name} thumbnail served")

    if not LIVE:
        after = httpx.get(f"{MOCK}/stats").json()
        n = after["workers_ai"] - mock_before["workers_ai"]
        check(n == 3, f"gateway saw 3 page requests (2 scan + 1 photo, none for born-digital): {n}")
        check(set(after["models"]) == {"@cf/deepseek-ai/deepseek-v4.1-flash-1017"}, f"default model used: {set(after['models'])}")
        check(after["aig"] >= 3, "cf-aig-authorization sent")
        check(all(ua.startswith("paperless-cf-ocr/") for ua in after["user_agents"]), "User-Agent set")

    print("\nLogs:")
    log = logs()
    check("Loaded third-party parser 'Cloudflare AI Gateway OCR'" in log, "Paperless discovered the plugin")
    check("Cloudflare OCR configured:" in log, "plugin configuration logged")
    check("Cloudflare OCR: 2 page(s)" in log, "per-document summary logged")
    # Every page must come from the gateway: a Tesseract fallback would still produce
    # plausible text, so successful content alone does not prove the gateway worked.
    summaries = re.findall(r"Cloudflare OCR: (\d+) page\(s\), .*? (\d+) failed", log)
    check(bool(summaries) and all(failed == "0" for _, failed in summaries),
          f"no page fell back to Tesseract (pages, failed per document: {summaries})")
    gw_errors = re.findall(r"HTTP \d{3} from AI Gateway[^\n]{0,120}", log)
    check(not gw_errors, f"no gateway errors: {gw_errors[:2]}")
    for line in log.splitlines():
        if "cf_ocr" in line or "third-party parser" in line or "paperless-cf-ocr" in line:
            print("   |", line[-220:])

    if failures:
        sys.exit(f"\n{len(failures)} check(s) failed")
    print("\nAll end-to-end checks passed")


if __name__ == "__main__":
    main()
