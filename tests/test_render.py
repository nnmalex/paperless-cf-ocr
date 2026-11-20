from __future__ import annotations

import io

from PIL import Image

from paperless_cf_ocr import render


def _open(b: bytes) -> Image.Image:
    im = Image.open(io.BytesIO(b))
    im.load()
    return im


def test_scan_detection(samples) -> None:
    assert render.scanned_page_ratio(samples["scan_pdf"]) == 1.0
    assert render.scanned_page_ratio(samples["digital_pdf"]) == 0.0


def test_pdf_pages_respect_dpi_max_px_and_limit(samples) -> None:
    pages = list(render.pdf_pages(samples["scan_pdf"], dpi=200, max_px=1000, limit=5))
    assert len(pages) == 2
    for b in pages:
        im = _open(b)
        assert im.format == "JPEG" and max(im.size) <= 1000
    assert len(list(render.pdf_pages(samples["scan_pdf"], 200, 1000, limit=1))) == 1
    assert render.pdf_page_count(samples["scan_pdf"]) == 2


def test_multipage_tiff_yields_each_frame(samples) -> None:
    pages = list(render.image_pages(samples["tiff"], max_px=800, limit=10))
    assert len(pages) == 3
    assert len(list(render.image_pages(samples["tiff"], 800, limit=2))) == 2


def test_alpha_png_flattened_and_downscaled(samples) -> None:
    (b,) = list(render.image_pages(samples["png"], max_px=2048, limit=5))
    im = _open(b)
    assert im.mode == "RGB" and im.size == (2048, 683)
    assert im.getpixel((10, 10)) == (255, 255, 255)  # transparent -> white, not black


def test_exif_orientation_applied() -> None:
    im = Image.new("RGB", (400, 200), "white")
    exif = im.getexif()
    exif[0x0112] = 6  # rotate 90° CW on display
    buf = io.BytesIO()
    im.save(buf, "JPEG", exif=exif)
    out = _open(render.encode_jpeg(Image.open(io.BytesIO(buf.getvalue())), 2048))
    assert out.size == (200, 400)
