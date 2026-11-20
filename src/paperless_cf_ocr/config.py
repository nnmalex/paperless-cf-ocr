"""Configuration resolution for the Cloudflare AI Gateway OCR parser.

Precedence, per setting:

1. ``PAPERLESS_CF_OCR_*`` environment variables (Paperless loads ``paperless.conf`` into
   the environment, so this covers native and Docker installs alike).
2. Paperless-ngx's own AI settings (``paperless.config.AIConfig``: UI/database values
   override ``PAPERLESS_AI_LLM_*`` env vars), but only when they point at Cloudflare
   AI Gateway.

This module is deliberately free of Django imports so it can be unit-tested; the
Paperless lookup lives in :func:`paperless_ai_settings`.
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol
from urllib.parse import urlparse

logger = logging.getLogger("paperless.parsing.cf_ocr")

ENV_PREFIX = "PAPERLESS_CF_OCR_"
GATEWAY_HOST = "gateway.ai.cloudflare.com"
DEFAULT_MODEL = "@cf/qwen/qwen3.8-27b"

# Chosen in milestone-1 evaluation (tools/lab): without these rules vision models
# "reconstruct" unreadable text from what similar forms usually say.
DEFAULT_PROMPT = """\
You are an OCR engine. Transcribe the text that is visibly present in this document image.

Rules:
- Copy text exactly as written, character for character, in natural reading order. Include handwriting.
- Never guess, complete, correct, or reconstruct text from what similar documents usually say. Only write what you can actually see.
- If a word or passage is too small, faint, or blurry to read with confidence, write [illegible] in its place instead of guessing.
- Ignore faint mirror-image text that shows through from the other side of the paper.
- For signatures write [signature]. Skip logos, barcodes and QR codes.
- Render tables as rows with cells separated by " | ". Keep form labels next to their values.
- Output only the transcription, no commentary."""

# Gateway path segments (after /v1/{account}/{gateway}/) whose API key is a Cloudflare
# token and can therefore be reused for Workers AI. Provider routes such as /openai
# carry that provider's key, which cannot call Workers AI.
_REUSABLE_KEY_ROUTES = {"compat", "workers-ai"}

_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_ACCOUNT_RE = re.compile(r"^[0-9a-f]{32}$")


class AISettings(Protocol):
    """The subset of ``paperless.config.AIConfig`` this plugin reads."""

    llm_backend: str | None
    llm_endpoint: str | None
    llm_api_key: str | None


@dataclass(frozen=True)
class GatewayRef:
    account_id: str
    gateway_id: str
    route: str | None = None  # first path segment after the gateway id, if any

    @property
    def base_url(self) -> str:
        return f"https://{GATEWAY_HOST}/v1/{self.account_id}/{self.gateway_id}"


@dataclass(frozen=True)
class Settings:
    gateway: GatewayRef
    api_token: str | None
    gateway_token: str | None
    model: str = DEFAULT_MODEL
    # Optional: re-run low-confidence pages on this model and use its output instead,
    # e.g. "anthropic/claude-sonnet-5-5". None disables the second opinion.
    second_opinion_model: str | None = None
    low_confidence_ratio: float = 0.01  # page is low-confidence if >1% of words have p<0.5
    low_confidence_min: float = 0.2  # ...or any single word has p<0.2 (or [illegible] appears)
    max_tokens: int = 8192
    prompt: str = DEFAULT_PROMPT
    extra_params: dict[str, Any] = field(default_factory=dict)
    timeout: float = 180.0
    max_retries: int = 4
    concurrency: int = 4
    dpi: int = 200
    max_image_px: int = 2048
    max_pages: int = 200
    score: int = 30
    archive: str = "hybrid"  # "hybrid" | "none"
    fallback: str = "tesseract"  # "tesseract" | "fail"
    source: str = "plugin"  # where the connection settings came from, for logs

    def chat_completions_url(self, model: str) -> str:
        """Workers AI models (``@cf/...``) use the gateway's Workers AI route; anything else
        (``anthropic/...``, ``openai/...``) goes through the unified OpenAI-compatible route."""
        if is_workers_ai(model):
            return f"{self.gateway.base_url}/workers-ai/v1/chat/completions"
        return f"{self.gateway.base_url}/compat/chat/completions"

    def headers(self) -> dict[str, str]:
        """HTTP headers for a gateway request. Never log the return value."""
        from . import __version__

        h = {
            "Content-Type": "application/json",
            # Cloudflare rejects urllib/unknown default UAs with "error code: 1010".
            "User-Agent": f"paperless-cf-ocr/{__version__}",
        }
        if self.api_token:
            h["Authorization"] = f"Bearer {self.api_token}"
        if self.gateway_token:
            h["cf-aig-authorization"] = f"Bearer {self.gateway_token}"
        return h

    def describe(self) -> str:
        """Secret-free summary for logs."""
        return (
            f"gateway={self.gateway.gateway_id} model={self.model} "
            f"second_opinion={self.second_opinion_model or 'off'} source={self.source} "
            f"auth={'api-token' if self.api_token else ''}"
            f"{'+' if self.api_token and self.gateway_token else ''}"
            f"{'gateway-token' if self.gateway_token else ''}"
        )


@dataclass
class Resolution:
    settings: Settings | None
    problems: list[str] = field(default_factory=list)
    enabled: bool = True

    @property
    def ok(self) -> bool:
        return self.enabled and self.settings is not None


def is_workers_ai(model: str) -> bool:
    return model.startswith(("@cf/", "@hf/"))


def parse_gateway_url(url: str | None) -> GatewayRef | None:
    """Parse ``https://gateway.ai.cloudflare.com/v1/{account}/{gateway}[/route/...]``."""
    if not url:
        return None
    u = urlparse(url.strip())
    if u.scheme != "https" or (u.hostname or "").lower() != GATEWAY_HOST:
        return None
    parts = [p for p in u.path.split("/") if p]
    if len(parts) < 3 or parts[0] != "v1":
        return None
    account, gateway = parts[1], parts[2]
    if not _ACCOUNT_RE.match(account) or not _ID_RE.match(gateway):
        return None
    return GatewayRef(account, gateway, parts[3] if len(parts) > 3 else None)


def resolve(env: Mapping[str, str], ai: AISettings | None = None) -> Resolution:
    """Work out the effective settings. Never raises; problems are reported instead."""
    problems: list[str] = []

    def get(name: str) -> str | None:
        v = env.get(ENV_PREFIX + name)
        return v.strip() if v and v.strip() else None

    def get_int(name: str, default: int, lo: int, hi: int) -> int:
        raw = get(name)
        if raw is None:
            return default
        try:
            v = int(raw)
        except ValueError:
            problems.append(f"{ENV_PREFIX}{name}={raw!r} is not an integer; using {default}")
            return default
        if not lo <= v <= hi:
            problems.append(f"{ENV_PREFIX}{name}={v} outside {lo}..{hi}; using {default}")
            return default
        return v

    def get_float(name: str, default: float, lo: float, hi: float) -> float:
        raw = get(name)
        if raw is None:
            return default
        try:
            v = float(raw)
        except ValueError:
            problems.append(f"{ENV_PREFIX}{name}={raw!r} is not a number; using {default}")
            return default
        if not lo <= v <= hi:
            problems.append(f"{ENV_PREFIX}{name}={v} outside {lo}..{hi}; using {default}")
            return default
        return v

    def get_choice(name: str, default: str, choices: tuple[str, ...]) -> str:
        raw = (get(name) or default).lower()
        if raw not in choices:
            problems.append(f"{ENV_PREFIX}{name}={raw!r} must be one of {choices}; using {default!r}")
            return default
        return raw

    enabled_raw = (get("ENABLED") or "true").lower()
    if enabled_raw in ("0", "false", "no", "off"):
        return Resolution(None, problems, enabled=False)

    # --- where is the gateway ---------------------------------------------------------
    source = "plugin"
    gateway: GatewayRef | None = None
    if url := get("GATEWAY_URL"):
        gateway = parse_gateway_url(url)
        if gateway is None:
            problems.append(
                f"{ENV_PREFIX}GATEWAY_URL must look like "
                f"https://{GATEWAY_HOST}/v1/<account-id>/<gateway-id>",
            )
    elif get("ACCOUNT_ID") or get("GATEWAY_ID"):
        acc, gw = get("ACCOUNT_ID"), get("GATEWAY_ID")
        if not (acc and gw):
            problems.append(f"set both {ENV_PREFIX}ACCOUNT_ID and {ENV_PREFIX}GATEWAY_ID")
        elif not _ACCOUNT_RE.match(acc):
            problems.append(f"{ENV_PREFIX}ACCOUNT_ID must be a 32-character hex account id")
        elif not _ID_RE.match(gw):
            problems.append(f"{ENV_PREFIX}GATEWAY_ID contains invalid characters")
        else:
            gateway = GatewayRef(acc, gw)

    ai_gateway: GatewayRef | None = None
    if ai is not None and (ai.llm_backend or "").lower() == "openai-like":
        ai_gateway = parse_gateway_url(ai.llm_endpoint)

    if gateway is None and not problems and ai_gateway is not None:
        gateway, source = GatewayRef(ai_gateway.account_id, ai_gateway.gateway_id), "paperless-ai"

    # --- credentials ------------------------------------------------------------------
    api_token = get("API_TOKEN")
    gateway_token = get("GATEWAY_TOKEN")
    if api_token is None and ai_gateway is not None and ai is not None and ai.llm_api_key:
        same_gateway = gateway is not None and (
            (gateway.account_id, gateway.gateway_id) == (ai_gateway.account_id, ai_gateway.gateway_id)
        )
        if same_gateway and ai_gateway.route in _REUSABLE_KEY_ROUTES:
            api_token = ai.llm_api_key
            source = "paperless-ai" if source == "paperless-ai" else "plugin+paperless-ai-key"
        elif same_gateway:
            problems.append(
                f"Paperless AI uses the gateway's '{ai_gateway.route}' provider route, so its API "
                f"key is not a Cloudflare token; set {ENV_PREFIX}API_TOKEN",
            )

    if gateway is None:
        if not problems:
            problems.append(
                f"no Cloudflare AI Gateway configured: set {ENV_PREFIX}ACCOUNT_ID and "
                f"{ENV_PREFIX}GATEWAY_ID (or {ENV_PREFIX}GATEWAY_URL), or point Paperless AI "
                f"(PAPERLESS_AI_LLM_ENDPOINT) at https://{GATEWAY_HOST}/v1/<account>/<gateway>/compat",
            )
        return Resolution(None, problems)
    if not (api_token or gateway_token):
        problems.append(f"no credentials: set {ENV_PREFIX}API_TOKEN (and {ENV_PREFIX}GATEWAY_TOKEN "
                        "if your gateway uses a separate authentication token)")
        return Resolution(None, problems)

    # --- request tuning ---------------------------------------------------------------
    extra: dict[str, Any] = {}
    if raw := get("EXTRA_PARAMS"):
        try:
            parsed = json.loads(raw)
        except ValueError:
            parsed = None
        if isinstance(parsed, dict):
            extra = parsed
        else:
            problems.append(f"{ENV_PREFIX}EXTRA_PARAMS must be a JSON object; ignoring it")

    prompt = get("PROMPT")
    if prompt is None and (path := get("PROMPT_FILE")):
        try:
            with open(path, encoding="utf-8") as fh:
                prompt = fh.read().strip() or None
        except OSError as e:
            problems.append(f"cannot read {ENV_PREFIX}PROMPT_FILE: {e}; using the default prompt")

    timeout = get_int("TIMEOUT", 180, 10, 3600)
    settings = Settings(
        gateway=gateway,
        api_token=api_token,
        gateway_token=gateway_token,
        model=get("MODEL") or DEFAULT_MODEL,
        second_opinion_model=get("SECOND_OPINION_MODEL"),
        low_confidence_ratio=get_float("LOW_CONFIDENCE_RATIO", 0.01, 0.0, 1.0),
        low_confidence_min=get_float("LOW_CONFIDENCE_MIN", 0.2, 0.0, 1.0),
        max_tokens=get_int("MAX_TOKENS", 8192, 256, 65536),
        prompt=prompt or DEFAULT_PROMPT,
        extra_params=extra,
        timeout=float(timeout),
        max_retries=get_int("MAX_RETRIES", 4, 0, 10),
        concurrency=get_int("CONCURRENCY", 4, 1, 32),
        dpi=get_int("DPI", 200, 72, 600),
        max_image_px=get_int("MAX_IMAGE_PX", 2048, 512, 8192),
        max_pages=get_int("MAX_PAGES", 200, 1, 10000),
        score=get_int("SCORE", 30, 1, 1000),
        archive=get_choice("ARCHIVE", "hybrid", ("hybrid", "none")),
        fallback=get_choice("FALLBACK", "tesseract", ("tesseract", "fail")),
        source=source,
    )
    return Resolution(settings, problems)


def paperless_ai_settings() -> AISettings | None:
    """Paperless's effective AI settings (DB/UI overrides applied), or None outside Paperless."""
    try:
        from paperless.config import AIConfig  # type: ignore[import-not-found]

        return AIConfig()
    except Exception:
        logger.debug("Paperless AI settings unavailable", exc_info=True)
        return None


_last_logged: tuple[str, ...] | None = None


def load() -> Resolution:
    """Resolve from the live environment and Paperless; log problems once per change."""
    global _last_logged
    res = resolve(os.environ, paperless_ai_settings())
    key = (str(res.ok), *(res.problems))
    if key != _last_logged:
        _last_logged = key
        if not res.enabled:
            logger.info("Cloudflare OCR parser disabled (%sENABLED=false)", ENV_PREFIX)
        for p in res.problems:
            logger.warning("Cloudflare OCR: %s", p)
        if res.settings:
            logger.info("Cloudflare OCR configured: %s", res.settings.describe())
    return res
