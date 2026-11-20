from __future__ import annotations

import subprocess

import httpx
import pikepdf
import pytest
import respx

from paperless_cf_ocr import client as cl
from paperless_cf_ocr import parser as p

from .conftest import OCR, PAGE_TEXT, ModeChoices

ACC = "0123456789abcdef0123456789abcdef"
WAI = f"https://gateway.ai.cloudflare.com/v1/{ACC}/gw/workers-ai/v1/chat/completions"


def norm(s: str) -> str:
    return " ".join(s.upper().split())


def pdftotext(path) -> str:
    return subprocess.run(["pdftotext", "-layout", str(path), "-"], capture_output=True,
                          text=True, check=True).stdout


@pytest.fixture
def env(monkeypatch):
    for k, v in {"ACCOUNT_ID": ACC, "GATEWAY_ID": "gw", "API_TOKEN": "t", "MAX_RETRIES": "0"}.items():
        monkeypatch.setenv(f"PAPERLESS_CF_OCR_{k}", v)
    monkeypatch.setattr(cl.time, "sleep", lambda s: None)


@pytest.fixture
def gw():
    with respx.mock(assert_all_called=False) as m:
        yield m


def ok(text: str) -> httpx.Response:
    return httpx.Response(200, json={"choices": [{"message": {"content": text},
                                                  "finish_reason": "stop"}]})


def run(path, mime, produce_archive):
    with p.CloudflareOcrParser() as parser:
        parser.parse(path, mime, produce_archive=produce_archive)
        archive = parser.get_archive_path()
        archive_text = pdftotext(archive) if archive else None
        is_pdfa = None
        if archive:
            with pikepdf.open(archive) as pdf:
                is_pdfa = "/OutputIntents" in pdf.Root
        thumb = parser.get_thumbnail(path, mime)
        return parser.get_text(), archive_text, is_pdfa, thumb.exists()


# --- the harness itself ---------------------------------------------------------------


def test_real_paperless_tesseract_parser_works(docs) -> None:
    from paperless.parsers.tesseract import RasterisedDocumentParser

    with RasterisedDocumentParser() as t:
        t.parse(docs["scan_pdf"], "application/pdf", produce_archive=True)
        assert t.get_archive_path() is not None
        assert "INVOICE 4711" in norm(t.get_text())


def test_plugin_sees_the_real_tesseract_parser(env) -> None:
    with p.CloudflareOcrParser() as parser:
        assert parser.can_produce_archive is True


# --- hybrid archive with real ocrmypdf/Tesseract --------------------------------------


def test_hybrid_scan_pdf(env, gw, docs) -> None:
    route = gw.post(WAI).mock(side_effect=[ok("LLM page one"), ok("LLM page two")])
    text, archive_text, is_pdfa, thumb = run(docs["scan_pdf"], "application/pdf", True)
    assert route.call_count == 2
    assert sorted(text.split("\n\n")) == ["LLM page one", "LLM page two"]  # content: LLM
    assert "INVOICE 4711" in norm(archive_text)  # archive: Tesseract text layer
    assert "THANK YOU" in norm(archive_text)
    assert is_pdfa is True and thumb


def test_hybrid_image(env, gw, docs) -> None:
    gw.post(WAI).mock(return_value=ok("LLM image text"))
    text, archive_text, is_pdfa, thumb = run(docs["png"], "image/png", True)
    assert text == "LLM image text"
    assert "INVOICE 4711" in norm(archive_text) and is_pdfa and thumb


def test_hybrid_all_pages_fail_uses_tesseract_text(env, gw, docs, monkeypatch) -> None:
    monkeypatch.setattr(p.shutil, "which", lambda name: None)  # disable per-page fallback
    gw.post(WAI).mock(return_value=httpx.Response(400, json={"error": {"message": "bad"}}))
    text, archive_text, _, _ = run(docs["scan_pdf"], "application/pdf", True)
    assert "INVOICE 4711" in norm(text) and archive_text


# --- decisions with real pdftotext / born-digital detection ---------------------------


def test_scan_with_existing_ocr_layer_is_reocrd(env, gw, docs) -> None:
    from paperless.parsers.utils import pdf_born_digital_text

    _, born = pdf_born_digital_text(docs["scan_with_layer_pdf"])
    assert born is True  # Paperless alone would keep the old layer
    route = gw.post(WAI).mock(return_value=ok("better LLM text"))
    text, archive_text, _, _ = run(docs["scan_with_layer_pdf"], "application/pdf", False)
    assert route.call_count == 2 and text == "better LLM text\n\nbetter LLM text"
    assert archive_text is None  # Paperless asks no archive for "born-digital" PDFs


def test_born_digital_pdf_keeps_its_text(env, gw, docs) -> None:
    route = gw.post(WAI).mock(return_value=ok("never"))
    text, archive_text, _, thumb = run(docs["digital_pdf"], "application/pdf", False)
    assert route.call_count == 0 and "BORN DIGITAL INVOICE LINE 24" in norm(text)
    assert archive_text is None and thumb


def test_ocr_mode_off_converts_without_ocr(env, gw, docs) -> None:
    OCR.mode = ModeChoices.OFF
    route = gw.post(WAI).mock(return_value=ok("never"))
    text, archive_text, _, _ = run(docs["png"], "image/png", True)
    assert route.call_count == 0 and text == ""
    assert archive_text is not None and not norm(archive_text)
    # Not asserting PDF/A here: Paperless v3.5.1's OCR_MODE=off image conversion calls
    # pikepdf's open_metadata() outside a `with` block, logs "PDF/A metadata stamping
    # failed" and falls back to a plain PDF (upstream bug, independent of this plugin).


# --- per-page local fallback with real Tesseract --------------------------------------


def test_page_fallback_with_real_tesseract(env, gw, docs) -> None:
    gw.post(WAI).mock(side_effect=[ok("LLM page"),
                                   httpx.Response(400, json={"error": {"message": "x"}})])
    import os

    os.environ["PAPERLESS_CF_OCR_CONCURRENCY"] = "1"
    try:
        text, _, _, _ = run(docs["scan_pdf"], "application/pdf", False)
    finally:
        del os.environ["PAPERLESS_CF_OCR_CONCURRENCY"]
    first, second = text.split("\n\n", 1)
    assert first == "LLM page" and "THANK YOU" in norm(second)
    assert PAGE_TEXT  # sanity
