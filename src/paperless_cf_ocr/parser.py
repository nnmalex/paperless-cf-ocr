"""Paperless-ngx parser plugin: OCR scanned documents with vision LLMs via Cloudflare AI Gateway.

Registered through the ``paperless_ngx.parsers`` entry point (see pyproject.toml).
"""

from __future__ import annotations

import concurrent.futures as cf
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Self

from django.conf import settings
from documents.parsers import ParseError, make_thumbnail_from_pdf
from paperless.parsers.utils import (
    extract_pdf_metadata,
    get_page_count_for_pdf,
    pdf_born_digital_text,
    post_process_text,
)

from . import __version__, config, render
from .client import GatewayClient, OcrError, PageResult

if TYPE_CHECKING:
    import datetime

    from paperless.parsers import MetadataEntry, ParserContext

logger = logging.getLogger("paperless.parsing.cf_ocr")

SUPPORTED_MIME_TYPES: dict[str, str] = {
    "application/pdf": ".pdf",
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/tiff": ".tiff",
    "image/bmp": ".bmp",
    "image/gif": ".gif",
    "image/webp": ".webp",
}

# Decisions for a PDF that may already carry text.
OCR, KEEP_LOCAL = "ocr", "keep-local"


def decide_pdf(ocr_mode: str, existing_text: str, born_digital: bool,
               scanned_ratio: float | None) -> str:
    """Whether to OCR a PDF or keep its embedded text.

    Paperless's ``PAPERLESS_OCR_MODE`` wins (``off`` never OCRs; ``redo``/``force`` always
    do). Otherwise a PDF without real text is always OCR'd, and one with text follows
    ``PAPERLESS_CF_OCR_EXISTING_TEXT``. In ``auto`` a PDF whose pages are full-page scan
    images is OCR'd even if it has a (typically poor, scanner-made) text layer.
    """
    if ocr_mode == "off":
        return KEEP_LOCAL
    if ocr_mode in ("redo", "force") or not born_digital:
        return OCR
    if existing_text == "trust":
        return KEEP_LOCAL
    if existing_text == "ignore":
        return OCR
    return OCR if (scanned_ratio or 0.0) >= 0.5 else KEEP_LOCAL


def _ocr_settings() -> tuple[str, str, int | None]:
    """Paperless's effective OCR mode, language and page limit (UI overrides env)."""
    try:
        from paperless.config import OcrConfig

        oc = OcrConfig()
        return str(getattr(oc.mode, "value", oc.mode)), oc.language or "eng", oc.pages or None
    except Exception:  # noqa: BLE001 - DB not ready etc.; fall back to settings
        return (str(getattr(settings, "OCR_MODE", "auto")),
                getattr(settings, "OCR_LANGUAGE", "eng") or "eng",
                getattr(settings, "OCR_PAGES", None) or None)


def _tesseract_parser() -> type | None:
    """Paperless's built-in Tesseract/ocrmypdf parser, used to build hybrid archives."""
    try:
        from paperless.parsers.tesseract import RasterisedDocumentParser
    except Exception:  # noqa: BLE001 - not inside Paperless, or an incompatible version
        return None
    return RasterisedDocumentParser


def _apply_pixel_limit() -> None:
    """Honour PAPERLESS_MAX_IMAGE_PIXELS for large scans, as the Tesseract parser does."""
    try:
        from documents.utils import maybe_override_pixel_limit

        maybe_override_pixel_limit()
    except Exception:
        logger.debug("could not apply Paperless image pixel limit", exc_info=True)


def tesseract_text(image: bytes, language: str, workdir: Path) -> str | None:
    """Local Tesseract fallback for a page the gateway could not transcribe."""
    exe = shutil.which("tesseract")
    if exe is None:
        return None
    src = workdir / "fallback.jpg"
    src.write_bytes(image)
    try:
        proc = subprocess.run(
            [exe, str(src), "stdout", "-l", language],
            capture_output=True, timeout=300, check=True,
        )
    except (subprocess.SubprocessError, OSError) as e:
        logger.warning("Tesseract fallback failed: %s", e)
        return None
    return proc.stdout.decode("utf-8", errors="replace")


class CloudflareOcrParser:
    name = "Cloudflare AI Gateway OCR"
    version = __version__
    author = "paperless-cf-ocr contributors"
    url = "https://github.com/nnmalex/paperless-cf-ocr"
    # Document images leave the server: respect PAPERLESS_REMOTE_OCR_MODE=workflow_only.
    uses_remote_service = True

    @classmethod
    def supported_mime_types(cls) -> dict[str, str]:
        return SUPPORTED_MIME_TYPES

    @classmethod
    def score(cls, mime_type: str, filename: str, path: Path | None = None) -> int | None:
        if mime_type not in SUPPORTED_MIME_TYPES:
            return None
        res = config.load()
        return res.settings.score if res.ok else None

    @property
    def can_produce_archive(self) -> bool:
        """Hybrid mode: Paperless's own Tesseract pipeline builds the searchable archive PDF
        (positioned text layer, PDF/A, deskew/rotate settings), while the document content
        comes from the vision model."""
        res = config.load()
        return bool(res.ok and res.settings.archive == "hybrid" and _tesseract_parser() is not None)

    @property
    def requires_pdf_rendition(self) -> bool:
        return False

    # -- lifecycle -------------------------------------------------------------------------

    def __init__(self, logging_group: object = None) -> None:
        settings.SCRATCH_DIR.mkdir(parents=True, exist_ok=True)
        self._tempdir = Path(tempfile.mkdtemp(prefix="paperless-cfocr-", dir=settings.SCRATCH_DIR))
        self._logging_group = logging_group
        self._text: str = ""
        self._archive_path: Path | None = None
        self._context: ParserContext | None = None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc_val: BaseException | None,
                 exc_tb: TracebackType | None) -> None:
        shutil.rmtree(self._tempdir, ignore_errors=True)

    def configure(self, context: ParserContext) -> None:
        self._context = context

    # -- parsing ---------------------------------------------------------------------------

    def parse(self, document_path: Path, mime_type: str, *, produce_archive: bool = True) -> None:
        res = config.load()
        if not res.ok:
            raise ParseError("Cloudflare OCR is not configured: " + "; ".join(res.problems))
        s = res.settings
        ocr_mode, language, ocr_pages = _ocr_settings()
        limit = min(s.max_pages, ocr_pages) if ocr_pages else s.max_pages
        want_archive = produce_archive and s.archive == "hybrid"

        pages: list[bytes] = []
        local_text = ""
        if mime_type == "application/pdf":
            local, born_digital = pdf_born_digital_text(document_path, log=logger)
            local_text = local or ""
            ratio = render.scanned_page_ratio(document_path) if born_digital else None
            decision = decide_pdf(ocr_mode, s.existing_text, born_digital, ratio)
            logger.debug("PDF text decision: %s (ocr_mode=%s existing_text=%s born_digital=%s "
                         "scanned_ratio=%s)", decision, ocr_mode, s.existing_text, born_digital,
                         ratio)
            if decision == OCR:
                total = render.pdf_page_count(document_path)
                pages = list(render.pdf_pages(document_path, s.dpi, s.max_image_px, limit))
        elif ocr_mode == "off":
            logger.info("PAPERLESS_OCR_MODE=off: not running OCR on image document")
        else:
            _apply_pixel_limit()
            pages = list(render.image_pages(document_path, s.max_image_px, limit))
            total = len(pages)

        if pages and total > len(pages):
            logger.warning("Only the first %d of %d pages are OCR'd (PAPERLESS_CF_OCR_MAX_PAGES/"
                           "PAPERLESS_OCR_PAGES)", len(pages), total)

        # Gateway OCR runs in the background (I/O bound) while the archive is built on this
        # thread: ocrmypdf uses multiprocessing and signal handlers, so it must not be
        # started from a worker thread.
        with cf.ThreadPoolExecutor(max_workers=1, thread_name_prefix="cf-ocr") as ex:
            ocr_future = ex.submit(self._ocr, pages, s, language) if pages else None
            hybrid_text = self._build_archive(document_path, mime_type) if want_archive else None
            outcome = ocr_future.result() if ocr_future else None

        if outcome is None:  # nothing to OCR: born-digital/trusted text, or OCR_MODE=off
            self._text = local_text
            return
        text, all_failed, first_error = outcome
        if all_failed:
            if hybrid_text and hybrid_text.strip():
                logger.warning("Cloudflare OCR failed on every page (%s); using the Tesseract "
                               "text from the archive pass", first_error)
                text = post_process_text(hybrid_text) or ""
            elif not text.strip():
                raise ParseError(f"Cloudflare OCR failed on every page: {first_error}")
        self._text = text

    def _build_archive(self, document_path: Path, mime_type: str) -> str | None:
        """Run Paperless's Tesseract parser for its searchable archive PDF.

        Returns Tesseract's text (used only if the gateway fails on every page). Archive
        problems never fail the document: Paperless then simply shows the original.
        """
        tesseract_cls = _tesseract_parser()
        if tesseract_cls is None:
            logger.warning("Hybrid archive requested but Paperless's Tesseract parser is "
                           "unavailable; no archive file")
            return None
        try:
            with tesseract_cls(self._logging_group) as t:
                if self._context is not None:
                    t.configure(self._context)
                t.parse(Path(document_path), mime_type, produce_archive=True)
                produced = t.get_archive_path()
                if produced and Path(produced).exists():
                    dest = self._tempdir / "archive.pdf"
                    shutil.copyfile(produced, dest)  # t's tempdir is removed on exit
                    self._archive_path = dest
                    logger.debug("Hybrid archive built by Tesseract: %s", dest)
                else:
                    logger.info("Tesseract produced no archive file for this document")
                return t.get_text()
        except Exception as e:  # noqa: BLE001 - archive is best-effort
            logger.warning("Hybrid archive generation failed (%s: %s); keeping the original "
                           "for display", type(e).__name__, e)
            return None

    def _ocr(self, pages: list[bytes], s: config.Settings,
             language: str) -> tuple[str, bool, OcrError | None]:
        """OCR pages; returns (text, every page failed, first error)."""
        results: list[PageResult | OcrError]
        try:
            with GatewayClient(s) as gw:
                results = gw.ocr_pages([(p, "image/jpeg") for p in pages])
        except OcrError as e:  # AuthError: affects every page
            logger.error("Cloudflare OCR failed for the whole document: %s", e)
            results = [e] * len(pages)

        texts: list[str] = []
        failed: list[int] = []
        low = second = 0
        for i, r in enumerate(results):
            if isinstance(r, PageResult):
                texts.append(r.text)
                low += r.low_confidence
                second += r.second_opinion_used
                continue
            failed.append(i + 1)
            if s.fallback == "fail":
                raise ParseError(f"Cloudflare OCR failed on page {i + 1}: {r}")
            fb = tesseract_text(pages[i], language, self._tempdir)
            if fb is None:
                logger.error("Page %d: Cloudflare OCR failed (%s) and no Tesseract fallback is "
                             "available; page text is empty", i + 1, r)
                texts.append("")
            else:
                logger.warning("Page %d: Cloudflare OCR failed (%s); used Tesseract", i + 1, r)
                texts.append(fb)

        logger.info("Cloudflare OCR: %d page(s), %d low-confidence, %d second opinion(s), "
                    "%d failed%s", len(pages), low, second, len(failed),
                    f" {failed}" if failed else "")
        first_error = next((r for r in results if isinstance(r, OcrError)), None)
        return (post_process_text("\n\n".join(texts)) or "",
                bool(failed) and len(failed) == len(pages), first_error)

    # -- results ---------------------------------------------------------------------------

    def get_text(self) -> str:
        return self._text

    def get_date(self) -> datetime.datetime | None:
        return None

    def get_archive_path(self) -> Path | None:
        return self._archive_path

    def get_thumbnail(self, document_path: Path, mime_type: str) -> Path:
        if self._archive_path is None and mime_type.startswith("image/"):
            out = self._tempdir / "thumb.webp"
            try:  # Paperless >= 3.4
                from documents.parsers import encode_thumbnail_webp
            except ImportError:
                render.image_thumbnail(Path(document_path), out)
            else:
                encode_thumbnail_webp(Path(document_path), out)
            return out
        return make_thumbnail_from_pdf(self._archive_path or Path(document_path), self._tempdir,
                                       self._logging_group)

    def get_page_count(self, document_path: Path, mime_type: str) -> int | None:
        if mime_type != "application/pdf":
            return None
        return get_page_count_for_pdf(Path(document_path), log=logger)

    def extract_metadata(self, document_path: Path, mime_type: str) -> list[MetadataEntry]:
        if mime_type != "application/pdf":
            return []
        return extract_pdf_metadata(Path(document_path), log=logger)
