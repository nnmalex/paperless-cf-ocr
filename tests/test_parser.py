from __future__ import annotations

import json
from importlib.metadata import entry_points
from typing import ClassVar

import httpx
import pytest
import respx

from paperless_cf_ocr import client as cl
from paperless_cf_ocr import parser as p

from .conftest import ParseError

ACC = "0123456789abcdef0123456789abcdef"
BASE = f"https://gateway.ai.cloudflare.com/v1/{ACC}/gw"
WAI = f"{BASE}/workers-ai/v1/chat/completions"


@pytest.fixture
def env(monkeypatch):
    for k, v in {"ACCOUNT_ID": ACC, "GATEWAY_ID": "gw", "API_TOKEN": "t", "MAX_RETRIES": "0"}.items():
        monkeypatch.setenv(f"PAPERLESS_CF_OCR_{k}", v)
    monkeypatch.setattr(cl.time, "sleep", lambda s: None)
    return monkeypatch


def ok(text: str) -> httpx.Response:
    return httpx.Response(200, json={"choices": [{"message": {"content": text},
                                                  "finish_reason": "stop"}]})


@pytest.fixture
def gw():
    with respx.mock(assert_all_called=False) as m:
        yield m


def page_texts(route) -> int:
    return route.call_count


# --- registration and scoring ---------------------------------------------------------


def test_entry_point_registered() -> None:
    eps = entry_points(group="paperless_ngx.parsers")
    ep = next(e for e in eps if e.name == "cf_ocr")
    assert ep.load() is p.CloudflareOcrParser


def test_identity_and_flags(paperless) -> None:
    cls = p.CloudflareOcrParser
    assert cls.name and cls.version and cls.author and cls.url
    assert cls.uses_remote_service is True
    with cls() as parser:
        assert parser.can_produce_archive is False and parser.requires_pdf_rendition is False


def test_score(paperless, env) -> None:
    assert p.CloudflareOcrParser.score("application/pdf", "x.pdf") == 30
    assert p.CloudflareOcrParser.score("text/plain", "x.txt") is None
    env.setenv("PAPERLESS_CF_OCR_SCORE", "25")
    assert p.CloudflareOcrParser.score("image/png", "x.png") == 25


def test_score_none_when_unconfigured_or_disabled(paperless, monkeypatch) -> None:
    for k in list(__import__("os").environ):
        if k.startswith("PAPERLESS_CF_OCR_"):
            monkeypatch.delenv(k)
    assert p.CloudflareOcrParser.score("application/pdf", "x.pdf") is None
    monkeypatch.setenv("PAPERLESS_CF_OCR_ACCOUNT_ID", ACC)
    monkeypatch.setenv("PAPERLESS_CF_OCR_GATEWAY_ID", "gw")
    monkeypatch.setenv("PAPERLESS_CF_OCR_API_TOKEN", "t")
    monkeypatch.setenv("PAPERLESS_CF_OCR_ENABLED", "false")
    assert p.CloudflareOcrParser.score("application/pdf", "x.pdf") is None


# --- PDF decision table ---------------------------------------------------------------


@pytest.mark.parametrize(("mode", "existing", "born", "ratio", "expected"), [
    ("off", "auto", False, None, p.KEEP_LOCAL),
    ("off", "ignore", True, 1.0, p.KEEP_LOCAL),
    ("redo", "trust", True, 0.0, p.OCR),
    ("force", "auto", True, 0.0, p.OCR),
    ("auto", "trust", False, None, p.OCR),  # no real text: always OCR
    ("auto", "trust", True, 1.0, p.KEEP_LOCAL),
    ("auto", "ignore", True, 0.0, p.OCR),
    ("auto", "auto", True, 1.0, p.OCR),  # scan with an OCR layer
    ("auto", "auto", True, 0.5, p.OCR),
    ("auto", "auto", True, 0.0, p.KEEP_LOCAL),  # born-digital
])
def test_decide_pdf(mode, existing, born, ratio, expected) -> None:
    assert p.decide_pdf(mode, existing, born, ratio) == expected


# --- parse ----------------------------------------------------------------------------


def run(path, mime, **kw):
    with p.CloudflareOcrParser() as parser:
        parser.parse(path, mime, **kw)
        return parser.get_text(), parser


def test_scanned_pdf_with_text_layer_is_ocrd(paperless, env, gw, samples) -> None:
    paperless.pdf_text, paperless.born_digital = "old poor scanner OCR " * 10, True
    route = gw.post(WAI).mock(side_effect=[ok("Page one  text"), ok("Page two")])
    text, parser = run(samples["scan_pdf"], "application/pdf")
    assert route.call_count == 2
    assert text in ("Page one text\n\nPage two", "Page two\n\nPage one text")
    assert parser.get_archive_path() is None and parser.get_date() is None


def test_pages_keep_document_order(paperless, env, gw, samples) -> None:
    def handler(request):
        url = json.loads(request.content)["messages"][0]["content"][1]["image_url"]["url"]
        return ok(f"len{len(url)}")
    gw.post(WAI).mock(side_effect=handler)
    text, _ = run(samples["scan_pdf"], "application/pdf")
    pages = list(p.render.pdf_pages(samples["scan_pdf"], 200, 2048, 10))
    import base64
    expected = [f"len{len('data:image/jpeg;base64,' + base64.b64encode(b).decode())}" for b in pages]
    assert text == "\n\n".join(expected)


def test_born_digital_pdf_keeps_local_text(paperless, env, gw, samples) -> None:
    paperless.pdf_text, paperless.born_digital = "Invoice line 0: 123.45 EUR", True
    route = gw.post(WAI).mock(return_value=ok("never"))
    text, _ = run(samples["digital_pdf"], "application/pdf")
    assert text == "Invoice line 0: 123.45 EUR" and route.call_count == 0


@pytest.mark.parametrize(("existing", "calls"), [("trust", 0), ("ignore", 2), ("auto", 2)])
def test_existing_text_setting(paperless, env, gw, samples, existing, calls) -> None:
    paperless.pdf_text, paperless.born_digital = "scanner text " * 10, True
    env.setenv("PAPERLESS_CF_OCR_EXISTING_TEXT", existing)
    route = gw.post(WAI).mock(return_value=ok("new"))
    run(samples["scan_pdf"], "application/pdf")
    assert route.call_count == calls


def test_ocr_mode_off_skips_ocr(paperless, env, gw, samples) -> None:
    paperless.ocr_mode, paperless.pdf_text = "off", "embedded"
    route = gw.post(WAI).mock(return_value=ok("never"))
    assert run(samples["scan_pdf"], "application/pdf")[0] == "embedded"
    assert run(samples["tiff"], "image/tiff")[0] == ""
    assert route.call_count == 0


def test_ocr_pages_limit(paperless, env, gw, samples) -> None:
    paperless.ocr_pages = 1
    route = gw.post(WAI).mock(return_value=ok("x"))
    run(samples["scan_pdf"], "application/pdf")
    assert route.call_count == 1


def test_scanned_pdf_ignores_produce_archive_flag(paperless, env, gw, samples) -> None:
    # Paperless says "no archive" for PDFs it thinks are born-digital (incl. scans with an
    # OCR layer) or with ARCHIVE_FILE_GENERATION=never; text must still be OCR'd.
    route = gw.post(WAI).mock(return_value=ok("x"))
    run(samples["scan_pdf"], "application/pdf", produce_archive=False)
    assert route.call_count == 2


def test_multipage_tiff(paperless, env, gw, samples) -> None:
    route = gw.post(WAI).mock(return_value=ok("frame"))
    text, _ = run(samples["tiff"], "image/tiff")
    assert route.call_count == 3 and text == "frame\n\nframe\n\nframe"


# --- failures and fallback ------------------------------------------------------------


def test_failed_page_without_tesseract_is_empty_but_document_survives(
        paperless, env, gw, samples, monkeypatch) -> None:
    monkeypatch.setattr(p.shutil, "which", lambda name: None)
    gw.post(WAI).mock(side_effect=[ok("good page"),
                                   httpx.Response(400, json={"error": {"message": "bad"}})])
    env.setenv("PAPERLESS_CF_OCR_CONCURRENCY", "1")
    text, _ = run(samples["scan_pdf"], "application/pdf")
    assert text == "good page"


def test_failed_page_uses_tesseract_fallback(paperless, env, gw, samples, monkeypatch) -> None:
    monkeypatch.setattr(p, "tesseract_text", lambda img, lang, wd: f"tesseract({lang})")
    paperless.ocr_language = "deu+eng"
    gw.post(WAI).mock(return_value=httpx.Response(400, json={"error": {"message": "bad"}}))
    text, _ = run(samples["tiff"], "image/tiff")
    assert text == "\n\n".join(["tesseract(deu+eng)"] * 3)


def test_fallback_fail_raises(paperless, env, gw, samples) -> None:
    env.setenv("PAPERLESS_CF_OCR_FALLBACK", "fail")
    gw.post(WAI).mock(return_value=httpx.Response(400, json={"error": {"message": "bad"}}))
    with pytest.raises(ParseError, match="page 1"):
        run(samples["tiff"], "image/tiff")


def test_every_page_failed_without_fallback_text_raises(paperless, env, gw, samples,
                                                         monkeypatch) -> None:
    monkeypatch.setattr(p.shutil, "which", lambda name: None)
    gw.post(WAI).mock(return_value=httpx.Response(400, json={"error": {"message": "bad"}}))
    with pytest.raises(ParseError, match="every page"):
        run(samples["tiff"], "image/tiff")


def test_auth_error_falls_back_for_whole_document(paperless, env, gw, samples,
                                                  monkeypatch) -> None:
    monkeypatch.setattr(p, "tesseract_text", lambda img, lang, wd: "local")
    route = gw.post(WAI).mock(return_value=httpx.Response(401, json={"errors": [{"message": "no"}]}))
    text, _ = run(samples["scan_pdf"], "application/pdf")
    assert text == "local\n\nlocal" and route.call_count <= 2


def test_parse_when_unconfigured_raises(paperless, monkeypatch, samples) -> None:
    monkeypatch.delenv("PAPERLESS_CF_OCR_ACCOUNT_ID", raising=False)
    with pytest.raises(ParseError, match="not configured"):
        run(samples["tiff"], "image/tiff")


# --- thumbnails, page count, metadata, cleanup ----------------------------------------


def test_thumbnail_page_count_metadata(paperless, env, samples) -> None:
    with p.CloudflareOcrParser() as parser:
        assert parser.get_thumbnail(samples["tiff"], "image/tiff").exists()
        assert parser.get_thumbnail(samples["scan_pdf"], "application/pdf").exists()
        assert paperless.thumbnails == [samples["tiff"], samples["scan_pdf"]]
        assert parser.get_page_count(samples["scan_pdf"], "application/pdf") == 99
        assert parser.get_page_count(samples["tiff"], "image/tiff") is None
        assert parser.extract_metadata(samples["scan_pdf"], "application/pdf") == ["meta"]
        assert parser.extract_metadata(samples["tiff"], "image/tiff") == []
        tmp = parser._tempdir
        assert tmp.exists()
    assert not tmp.exists()


# --- hybrid archive (milestone 5) -----------------------------------------------------


class FakeTesseract:
    """Stands in for paperless.parsers.tesseract.RasterisedDocumentParser."""

    calls: ClassVar[list[dict]] = []
    fail: Exception | None = None
    produce: bool = True
    text = "tesseract text"

    def __init__(self, logging_group=None) -> None:
        import tempfile
        import threading
        from pathlib import Path

        self.tmp = Path(tempfile.mkdtemp())
        self.rec = {"thread": threading.current_thread() is threading.main_thread(),
                    "context": None, "group": logging_group}
        self._archive = None

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        import shutil
        shutil.rmtree(self.tmp)

    def configure(self, context) -> None:
        self.rec["context"] = context

    def parse(self, path, mime, *, produce_archive=True) -> None:
        self.rec.update(path=path, mime=mime, produce_archive=produce_archive)
        FakeTesseract.calls.append(self.rec)
        if FakeTesseract.fail:
            raise FakeTesseract.fail
        if FakeTesseract.produce:
            self._archive = self.tmp / "archive.pdf"
            self._archive.write_bytes(b"%PDF-1.7 archive")

    def get_archive_path(self):
        return self._archive

    def get_text(self) -> str:
        return FakeTesseract.text


@pytest.fixture
def tess(monkeypatch):
    FakeTesseract.calls, FakeTesseract.fail, FakeTesseract.produce = [], None, True
    FakeTesseract.text = "tesseract text"
    monkeypatch.setattr(p, "_tesseract_parser", lambda: FakeTesseract)
    return FakeTesseract


def test_can_produce_archive_in_hybrid_mode(paperless, env, tess) -> None:
    with p.CloudflareOcrParser() as parser:
        assert parser.can_produce_archive is True
    env.setenv("PAPERLESS_CF_OCR_ARCHIVE", "none")
    with p.CloudflareOcrParser() as parser:
        assert parser.can_produce_archive is False


def test_can_produce_archive_false_without_tesseract(paperless, env) -> None:
    with p.CloudflareOcrParser() as parser:
        assert parser.can_produce_archive is False  # fake env has no tesseract parser


def test_hybrid_archive_with_llm_text(paperless, env, gw, tess, samples) -> None:
    gw.post(WAI).mock(return_value=ok("LLM text"))
    ctx = object()
    with p.CloudflareOcrParser("grp") as parser:
        parser.configure(ctx)
        parser.parse(samples["tiff"], "image/tiff", produce_archive=True)
        archive = parser.get_archive_path()
        assert parser.get_text() == "LLM text\n\nLLM text\n\nLLM text"  # content from the LLM
        assert archive is not None and archive.read_bytes() == b"%PDF-1.7 archive"
        assert archive.parent == parser._tempdir  # copied before Tesseract cleaned up
        parser.get_thumbnail(samples["tiff"], "image/tiff")
        assert paperless.thumbnails[-1] == archive  # thumbnail from the archive PDF
    (call,) = tess.calls
    assert call["thread"] is True  # ocrmypdf must run on the calling (main) thread
    assert call["produce_archive"] is True and call["context"] is ctx and call["group"] == "grp"


def test_no_archive_when_not_requested(paperless, env, gw, tess, samples) -> None:
    gw.post(WAI).mock(return_value=ok("x"))
    _, parser = run(samples["scan_pdf"], "application/pdf", produce_archive=False)
    assert tess.calls == [] and parser.get_archive_path() is None


def test_no_archive_when_archive_mode_none(paperless, env, gw, tess, samples) -> None:
    env.setenv("PAPERLESS_CF_OCR_ARCHIVE", "none")
    gw.post(WAI).mock(return_value=ok("x"))
    _, parser = run(samples["scan_pdf"], "application/pdf", produce_archive=True)
    assert tess.calls == [] and parser.get_archive_path() is None


def test_archive_failure_does_not_fail_document(paperless, env, gw, tess, samples) -> None:
    tess.fail = RuntimeError("ghostscript exploded")
    gw.post(WAI).mock(return_value=ok("still here"))
    text, parser = run(samples["scan_pdf"], "application/pdf", produce_archive=True)
    assert text == "still here\n\nstill here" and parser.get_archive_path() is None


def test_tesseract_without_archive_file(paperless, env, gw, tess, samples) -> None:
    tess.produce = False
    gw.post(WAI).mock(return_value=ok("x"))
    _, parser = run(samples["scan_pdf"], "application/pdf", produce_archive=True)
    assert parser.get_archive_path() is None


def test_born_digital_with_archive_always(paperless, env, gw, tess, samples) -> None:
    # ARCHIVE_FILE_GENERATION=always on a born-digital PDF: keep its text, still build archive.
    paperless.pdf_text, paperless.born_digital = "digital text", True
    route = gw.post(WAI).mock(return_value=ok("never"))
    text, parser = run(samples["digital_pdf"], "application/pdf", produce_archive=True)
    assert text == "digital text" and route.call_count == 0
    assert parser.get_archive_path() is not None


def test_ocr_mode_off_still_converts_archive(paperless, env, gw, tess, samples) -> None:
    paperless.ocr_mode = "off"
    route = gw.post(WAI).mock(return_value=ok("never"))
    text, parser = run(samples["tiff"], "image/tiff", produce_archive=True)
    assert text == "" and route.call_count == 0 and parser.get_archive_path() is not None


def test_all_pages_failed_uses_archive_tesseract_text(paperless, env, gw, tess, samples,
                                                      monkeypatch) -> None:
    monkeypatch.setattr(p.shutil, "which", lambda name: None)  # no per-page fallback
    gw.post(WAI).mock(return_value=httpx.Response(400, json={"error": {"message": "bad"}}))
    text, parser = run(samples["tiff"], "image/tiff", produce_archive=True)
    assert text == "tesseract text" and parser.get_archive_path() is not None


def test_fallback_fail_still_raises_in_hybrid(paperless, env, gw, tess, samples) -> None:
    env.setenv("PAPERLESS_CF_OCR_FALLBACK", "fail")
    gw.post(WAI).mock(return_value=httpx.Response(400, json={"error": {"message": "bad"}}))
    with pytest.raises(ParseError, match="page 1"):
        run(samples["tiff"], "image/tiff", produce_archive=True)


def test_image_thumbnail_on_paperless_before_3_4(paperless, env, samples, monkeypatch) -> None:
    import sys
    monkeypatch.delattr(sys.modules["documents.parsers"], "encode_thumbnail_webp")
    with p.CloudflareOcrParser() as parser:
        thumb = parser.get_thumbnail(samples["png"], "image/png")
        assert thumb.exists() and thumb.read_bytes()[8:12] == b"WEBP"
