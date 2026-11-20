"""Run the plugin against REAL Paperless-ngx parsing code plus real Tesseract/ocrmypdf.

Loaded unmodified from the Paperless tag fetched by ``fetch_paperless.sh``:
  paperless.parsers.tesseract (RasterisedDocumentParser), paperless.parsers.utils,
  documents.parsers, documents.utils
Stubbed (they need Django models / a database): django.conf.settings, paperless.models
choice enums, paperless.config.OcrConfig, paperless.version, documents.loggers,
paperless.parsers.registry. Stub values mirror Paperless v3.5.1 defaults.

Needs: tesseract, ghostscript, poppler (pdftotext/pdftoppm), unpaper, ocrmypdf.
Run: tests_integration/fetch_paperless.sh && pytest tests_integration
"""

from __future__ import annotations

import enum
import importlib.util
import logging
import shutil
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
PAPERLESS = HERE / "_paperless"
TOOLS = ("tesseract", "gs", "pdftotext", "pdftoppm", "unpaper")

missing = [t for t in TOOLS if shutil.which(t) is None]
if not (PAPERLESS / "paperless/parsers/tesseract.py").exists():
    pytest.skip("run tests_integration/fetch_paperless.sh first", allow_module_level=True)
if missing:
    pytest.skip(f"missing tools: {missing}", allow_module_level=True)
try:
    import ocrmypdf  # noqa: F401
except ImportError:
    pytest.skip("ocrmypdf not installed", allow_module_level=True)


class _TextChoices(str, enum.Enum):
    def __str__(self) -> str:
        return self.value


class ModeChoices(_TextChoices):
    AUTO = "auto"
    FORCE = "force"
    REDO = "redo"
    OFF = "off"


class CleanChoices(_TextChoices):
    CLEAN = "clean"
    FINAL = "clean-final"
    NONE = "none"


class OutputTypeChoices(_TextChoices):
    PDF = "pdf"
    PDF_A = "pdfa"
    PDF_A1 = "pdfa-1"
    PDF_A2 = "pdfa-2"
    PDF_A3 = "pdfa-3"


@dataclass
class OcrSettings:
    """Paperless v3.5.1 defaults; tests mutate ``OCR`` to emulate env/UI settings."""

    pages: int | None = None
    language: str = "eng"
    mode: ModeChoices = ModeChoices.AUTO
    archive_file_generation: str = "auto"
    image_dpi: int | None = None
    clean: CleanChoices = CleanChoices.CLEAN
    deskew: bool = True
    rotate: bool = True
    rotate_threshold: float = 12.0
    max_image_pixel: float | None = None
    color_conversion_strategy: str = "RGB"
    user_args: dict | None = None
    output_type: str = OutputTypeChoices.PDF_A.value
    extra: dict = field(default_factory=dict)


OCR = OcrSettings()


class OcrConfig:
    def __init__(self) -> None:
        for k, v in vars(OCR).items():
            setattr(self, k, v)


class _Settings:
    SCRATCH_DIR = Path("/nonexistent")
    THREADS_PER_WORKER = 1
    MAX_IMAGE_PIXELS = None
    OCR_MODE = "auto"
    OCR_LANGUAGE = "eng"
    OCR_PAGES = None


def _mod(name: str, **attrs: object) -> types.ModuleType:
    m = types.ModuleType(name)
    m.__path__ = []  # allow submodules
    for k, v in attrs.items():
        setattr(m, k, v)
    sys.modules[name] = m
    return m


def _load_real(name: str, rel: str) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, PAPERLESS / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    parent, _, child = name.rpartition(".")
    setattr(sys.modules[parent], child, mod)
    return mod


class LoggingMixin:
    logging_group = None

    @property
    def log(self) -> logging.Logger:
        return logging.getLogger("paperless.parsing")

    def renew_logging_group(self) -> None:
        pass


def _install() -> None:
    _mod("django")
    _mod("django.conf", settings=_Settings)
    _mod("documents")
    _mod("documents.loggers", LoggingMixin=LoggingMixin)
    _mod("paperless")
    _mod("paperless.parsers")
    _mod("paperless.version", __full_version_str__="3.5.1-test")
    _mod("paperless.models", ModeChoices=ModeChoices, CleanChoices=CleanChoices,
         OutputTypeChoices=OutputTypeChoices)
    _mod("paperless.config", OcrConfig=OcrConfig,
         OCR_RESERVED_USER_ARGS=frozenset({"input_file_or_options", "output_file", "sidecar"}))

    def _no_registry():  # only used by helpers the plugin does not call
        raise RuntimeError("parser registry is not available in integration tests")

    _mod("paperless.parsers.registry", get_parser_registry=_no_registry)
    _load_real("documents.utils", "documents/utils.py")
    _load_real("paperless.parsers.utils", "paperless/parsers/utils.py")
    _load_real("documents.parsers", "documents/parsers.py")
    _load_real("paperless.parsers.tesseract", "paperless/parsers/tesseract.py")


_install()


@pytest.fixture(autouse=True)
def paperless_env(tmp_path, monkeypatch):
    fresh = OcrSettings()
    for k, v in vars(fresh).items():
        setattr(OCR, k, v)
    monkeypatch.setattr(_Settings, "SCRATCH_DIR", tmp_path / "scratch")
    return OCR


# --- generated documents with text large enough for Tesseract ------------------------

PAGE_TEXT = ["SCANNED PAGE ONE Invoice 4711 Total 123.45 EUR",
             "SCANNED PAGE TWO Thank you for your business"]


def _page(text: str):
    from PIL import Image, ImageDraw, ImageFont

    im = Image.new("L", (1700, 2200), 255)  # US letter at 200 dpi
    font = ImageFont.load_default(size=56)
    d = ImageDraw.Draw(im)
    y = 200
    words = text.split()
    for chunk in (" ".join(words[:3]), " ".join(words[3:])):
        d.text((150, y), chunk, fill=0, font=font)
        y += 120
    return im


@pytest.fixture(scope="session")
def docs(tmp_path_factory) -> dict[str, Path]:
    from fpdf import FPDF

    d = tmp_path_factory.mktemp("docs")
    out: dict[str, Path] = {}
    pdf = FPDF(unit="pt", format="letter")
    for i, text in enumerate(PAGE_TEXT):
        img = d / f"p{i}.png"
        _page(text).save(img, dpi=(200, 200))
        pdf.add_page()
        pdf.image(str(img), x=0, y=0, w=pdf.w, h=pdf.h)
    out["scan_pdf"] = d / "scan.pdf"
    pdf.output(str(out["scan_pdf"]))

    out["png"] = d / "page.png"
    _page(PAGE_TEXT[0]).save(out["png"], dpi=(200, 200))

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("helvetica", size=12)
    for i in range(25):
        pdf.cell(0, 8, f"Born digital invoice line {i}: 99.99 EUR", new_x="LMARGIN",
                 new_y="NEXT")
    out["digital_pdf"] = d / "digital.pdf"
    pdf.output(str(out["digital_pdf"]))

    # A scan that already carries an OCR text layer (like many scanner outputs).
    import ocrmypdf

    out["scan_with_layer_pdf"] = d / "scan_with_layer.pdf"
    ocrmypdf.ocr(out["scan_pdf"], out["scan_with_layer_pdf"], output_type="pdf",
                 progress_bar=False)
    return out
