"""Turn documents into page images for the vision model, and inspect PDFs.

Uses only libraries every Paperless-ngx install ships (via ocrmypdf): pypdfium2 and Pillow.
No Django imports here, so this module is unit-testable on its own.
"""

from __future__ import annotations

import io
import logging
from collections.abc import Iterator
from pathlib import Path

from PIL import Image, ImageOps, ImageSequence

logger = logging.getLogger("paperless.parsing.cf_ocr")

JPEG_QUALITY = 90
# A page counts as scanned when one image covers at least this share of it.
SCAN_COVERAGE = 0.85
MULTI_FRAME_FORMATS = {"TIFF", "MPO"}  # animated GIF/WebP: only the first frame is a "page"


def encode_jpeg(im: Image.Image, max_px: int) -> bytes:
    """Normalise orientation/colour, downscale to ``max_px`` on the long edge, encode JPEG."""
    im = ImageOps.exif_transpose(im)
    if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
        rgba = im.convert("RGBA")
        flat = Image.new("RGB", rgba.size, (255, 255, 255))
        flat.paste(rgba, mask=rgba.split()[-1])
        im = flat
    elif im.mode in ("1", "L", "I", "I;16", "F"):
        im = im.convert("L")
    elif im.mode != "RGB":
        im = im.convert("RGB")
    if max(im.size) > max_px:
        im = im.copy()
        im.thumbnail((max_px, max_px), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=JPEG_QUALITY, optimize=True)
    return buf.getvalue()


THUMB_MAX = (500, 700)


def image_thumbnail(src: Path, out: Path) -> None:
    """WebP thumbnail for an image document (first frame, alpha flattened onto white).

    Fallback for Paperless < 3.4, whose ``documents.parsers`` lacks ``encode_thumbnail_webp``.
    """
    with Image.open(src) as im:
        im = ImageOps.exif_transpose(im)
        if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
            rgba = im.convert("RGBA")
            flat = Image.new("RGB", rgba.size, (255, 255, 255))
            flat.paste(rgba, mask=rgba.split()[-1])
            im = flat
        else:
            im = im.convert("RGB")
        im.thumbnail(THUMB_MAX, Image.Resampling.LANCZOS)
        im.save(out, format="WEBP")


def pdf_page_count(path: Path) -> int:
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(str(path))
    try:
        return len(pdf)
    finally:
        pdf.close()


def pdf_pages(path: Path, dpi: int, max_px: int, limit: int) -> Iterator[bytes]:
    """Render up to ``limit`` PDF pages to JPEG, one at a time (bounded memory)."""
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(str(path))
    try:
        for i in range(min(len(pdf), limit)):
            page = pdf[i]
            try:
                w_pt, h_pt = page.get_size()
                # Never render far beyond what we send: cap the scale at max_px.
                scale = min(dpi / 72.0, max_px / max(w_pt, h_pt, 1.0))
                bitmap = page.render(scale=scale, may_draw_forms=True)
                try:
                    yield encode_jpeg(bitmap.to_pil(), max_px)
                finally:
                    bitmap.close()
            finally:
                page.close()
    finally:
        pdf.close()


def image_pages(path: Path, max_px: int, limit: int) -> Iterator[bytes]:
    """Yield JPEG pages from an image file; multi-page TIFFs yield one per frame."""
    with Image.open(path) as im:
        frames = ImageSequence.Iterator(im) if im.format in MULTI_FRAME_FORMATS else [im]
        for n, frame in enumerate(frames):
            if n >= limit:
                break
            yield encode_jpeg(frame, max_px)


def scanned_page_ratio(path: Path, sample: int = 20) -> float:
    """Share of (up to ``sample``) pages that are dominated by one raster image.

    Distinguishes scans that carry an OCR text layer from born-digital PDFs: both have
    extractable text, but only scans place a full-page image on each page.
    """
    import pypdfium2 as pdfium
    import pypdfium2.raw as pdfium_c

    pdf = pdfium.PdfDocument(str(path))
    try:
        n = min(len(pdf), sample)
        if n == 0:
            return 0.0
        scanned = 0
        for i in range(n):
            page = pdf[i]
            try:
                # Image positions are in unrotated user space, so measure against the
                # (unrotated) crop box rather than get_size(), which reflects /Rotate.
                bl, bb, br, bt = page.get_cropbox()
                area = max((br - bl) * (bt - bb), 1.0)
                best = 0.0
                for obj in page.get_objects(filter=(pdfium_c.FPDF_PAGEOBJ_IMAGE,), max_depth=3):
                    # pypdfium2 >= 5 (required by ocrmypdf 17): get_bounds(); 4.x: get_pos().
                    bounds = getattr(obj, "get_bounds", None) or obj.get_pos
                    left, bottom, right, top = bounds()
                    # Clip to the page: scans are often placed with a little bleed.
                    cw = max(0.0, min(right, br) - max(left, bl))
                    ch = max(0.0, min(top, bt) - max(bottom, bb))
                    best = max(best, cw * ch / area)
                if best >= SCAN_COVERAGE:
                    scanned += 1
            finally:
                page.close()
        return scanned / n
    finally:
        pdf.close()
