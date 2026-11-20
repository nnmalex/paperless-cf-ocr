"""Paperless-ngx parser plugin: OCR scanned documents with vision LLMs via Cloudflare AI Gateway.

Registered through the ``paperless_ngx.parsers`` entry point (see pyproject.toml).
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Self

from django.conf import settings
from documents.parsers import ParseError, encode_thumbnail_webp, make_thumbnail_from_pdf
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
    url = "https://pypi.org/project/paperless-cf-ocr/"
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
        # Searchable archive PDFs arrive with the hybrid mode (milestone 5).
        return False

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

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc_val: BaseException | None,
                 exc_tb: TracebackType | None) -> None:
        shutil.rmtree(self._tempdir, ignore_errors=True)

    def configure(self, context: ParserContext) -> None:
        pass

    # -- parsing ---------------------------------------------------------------------------

    def parse(self, document_path: Path, mime_type: str, *, produce_archive: bool = True) -> None:
        res = config.load()
        if not res.ok:
            raise ParseError("Cloudflare OCR is not configured: " + "; ".join(res.problems))
        s = res.settings
        ocr_mode, language, ocr_pages = _ocr_settings()
        limit = min(s.max_pages, ocr_pages) if ocr_pages else s.max_pages

        if mime_type == "application/pdf":
            local_text, born_digital = pdf_born_digital_text(document_path, log=logger)
            ratio = render.scanned_page_ratio(document_path) if born_digital else None
            decision = decide_pdf(ocr_mode, s.existing_text, born_digital, ratio)
            logger.debug("PDF text decision: %s (ocr_mode=%s existing_text=%s born_digital=%s "
                         "scanned_ratio=%s)", decision, ocr_mode, s.existing_text, born_digital,
                         ratio)
            if decision == KEEP_LOCAL:
                self._text = local_text or ""
                return
            total = render.pdf_page_count(document_path)
            pages = list(render.pdf_pages(document_path, s.dpi, s.max_image_px, limit))
        else:
            if ocr_mode == "off":
                logger.info("PAPERLESS_OCR_MODE=off: not running OCR on image document")
                self._text = ""
                return
            _apply_pixel_limit()
            pages = list(render.image_pages(document_path, s.max_image_px, limit))
            total = len(pages)

        if total > len(pages):
            logger.warning("Only the first %d of %d pages are OCR'd (PAPERLESS_CF_OCR_MAX_PAGES/"
                           "PAPERLESS_OCR_PAGES)", len(pages), total)
        self._text = self._ocr(pages, s, language)

    def _ocr(self, pages: list[bytes], s: config.Settings, language: str) -> str:
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

        if failed and len(failed) == len(pages) and not any(t.strip() for t in texts):
            raise ParseError(f"Cloudflare OCR failed on every page: {results[0]}")
        logger.info("Cloudflare OCR: %d page(s), %d low-confidence, %d second opinion(s), "
                    "%d failed%s", len(pages), low, second, len(failed),
                    f" {failed}" if failed else "")
        return post_process_text("\n\n".join(texts)) or ""

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
