"""Minimal stand-ins for the Paperless-ngx modules the parser imports.

The parser runs inside Paperless (Django + DB). These fakes provide just the functions it
calls, with behaviour controllable from tests via the ``paperless`` fixture.
"""

from __future__ import annotations

import re
import sys
import types
from pathlib import Path

import pytest


class _Settings:
    SCRATCH_DIR = Path("/nonexistent")  # replaced per test
    OCR_MODE = "auto"
    OCR_LANGUAGE = "eng"
    OCR_PAGES = None


class _State:
    """Knobs the tests turn."""

    def __init__(self) -> None:
        self.ocr_mode = "auto"
        self.ocr_language = "eng"
        self.ocr_pages: int | None = None
        self.pdf_text: str | None = None
        self.born_digital = False
        self.thumbnails: list[Path] = []


STATE = _State()


class ParseError(Exception):
    pass


def _post_process_text(text: str | None) -> str | None:  # copy of Paperless's helper
    if not text:
        return None
    t = re.sub(r"([^\S\r\n]+)", " ", text)
    t = re.sub(r"([\n\r]+)([^\S\n\r]+)", "\\1", t)
    t = re.sub(r"([^\S\n\r]+)$", "", t)
    return t.strip().replace("\0", " ") or None


def _install() -> None:
    def mod(name: str, **attrs: object) -> types.ModuleType:
        m = sys.modules.get(name) or types.ModuleType(name)
        for k, v in attrs.items():
            setattr(m, k, v)
        sys.modules[name] = m
        return m

    mod("django")
    mod("django.conf", settings=_Settings)

    def encode_thumbnail_webp(src: Path, out: Path, **_: object) -> None:
        out.write_bytes(b"WEBP")
        STATE.thumbnails.append(src)

    def make_thumbnail_from_pdf(src: Path, tmp: Path, group: object = None) -> Path:
        out = tmp / "pdf-thumb.webp"
        out.write_bytes(b"WEBP")
        STATE.thumbnails.append(src)
        return out

    mod("documents")
    mod("documents.parsers", ParseError=ParseError, encode_thumbnail_webp=encode_thumbnail_webp,
        make_thumbnail_from_pdf=make_thumbnail_from_pdf)
    mod("documents.utils", maybe_override_pixel_limit=lambda: None)

    class OcrConfig:
        def __init__(self) -> None:
            self.mode = STATE.ocr_mode
            self.language = STATE.ocr_language
            self.pages = STATE.ocr_pages

    mod("paperless")
    mod("paperless.config", OcrConfig=OcrConfig)  # no AIConfig: plugin must cope
    mod("paperless.parsers")
    mod("paperless.parsers.utils",
        post_process_text=_post_process_text,
        pdf_born_digital_text=lambda path, log=None: (STATE.pdf_text, STATE.born_digital),
        get_page_count_for_pdf=lambda path, log=None: 99,
        extract_pdf_metadata=lambda path, log=None: ["meta"])


_install()


@pytest.fixture
def paperless(tmp_path, monkeypatch):
    fresh = _State()
    for k, v in vars(fresh).items():
        setattr(STATE, k, v)
    monkeypatch.setattr(_Settings, "SCRATCH_DIR", tmp_path / "scratch")
    return STATE


# --- generated sample documents ------------------------------------------------------


def _page_image(text: str, size=(1240, 1754)):
    from PIL import Image, ImageDraw

    im = Image.new("L", size, 255)
    ImageDraw.Draw(im).text((100, 100), text, fill=0)
    return im


@pytest.fixture(scope="session")
def samples(tmp_path_factory) -> dict[str, Path]:
    from fpdf import FPDF

    d = tmp_path_factory.mktemp("samples")
    out: dict[str, Path] = {}

    # "Scan": every page is one full-page raster image.
    pdf = FPDF(unit="pt", format="A4")
    for i in range(2):
        img = d / f"scan{i}.png"
        _page_image(f"SCANNED PAGE {i + 1}").save(img)
        pdf.add_page()
        pdf.image(str(img), x=0, y=0, w=pdf.w, h=pdf.h)
    out["scan_pdf"] = d / "scan.pdf"
    pdf.output(str(out["scan_pdf"]))

    # Born-digital: vector text with a small logo.
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("helvetica", size=12)
    for i in range(20):
        pdf.cell(0, 8, f"Invoice line {i}: 123.45 EUR", new_x="LMARGIN", new_y="NEXT")
    logo = d / "logo.png"
    _page_image("LOGO", (200, 80)).save(logo)
    pdf.image(str(logo), x=150, y=10, w=40)
    out["digital_pdf"] = d / "digital.pdf"
    pdf.output(str(out["digital_pdf"]))

    # Multi-page TIFF and a single PNG with an alpha channel.
    frames = [_page_image(f"TIFF {i}").convert("RGB") for i in range(3)]
    out["tiff"] = d / "multi.tiff"
    frames[0].save(out["tiff"], save_all=True, append_images=frames[1:])
    from PIL import Image

    rgba = Image.new("RGBA", (3000, 1000), (255, 255, 255, 0))
    out["png"] = d / "alpha.png"
    rgba.save(out["png"])
    return out
