"""HTTP client for vision-model OCR through Cloudflare AI Gateway.

Routes (see :meth:`Settings.chat_completions_url`):

* ``@cf/...`` models  -> ``{gateway}/workers-ai/v1/chat/completions`` with ``temperature: 0``
  and ``logprobs: true`` (for the page confidence score).
* anything else (``anthropic/claude-sonnet-5-5`` ...) -> ``{gateway}/compat/chat/completions``
  without sampling parameters (Claude models reject a non-default temperature).

Behaviour learned in the milestone-1 lab: Workers AI returns 408 after ~120 s under load and
reasoning models occasionally return an empty answer, so both are retried.
"""

from __future__ import annotations

import base64
import concurrent.futures as cf
import logging
import random
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Self

import httpx

from .confidence import PageConfidence, page_confidence
from .config import Settings, is_workers_ai

logger = logging.getLogger("paperless.parsing.cf_ocr")

RETRYABLE_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504, 520, 521, 522, 523, 524, 529})
MAX_BACKOFF = 30.0
MAX_RETRY_AFTER = 60.0


class OcrError(Exception):
    """A page could not be transcribed."""


class AuthError(OcrError):
    """Credentials or gateway authentication were rejected; retrying cannot help."""


@dataclass
class Transcription:
    text: str
    model: str
    confidence: PageConfidence | None
    finish_reason: str | None
    usage: dict[str, Any] = field(default_factory=dict)
    seconds: float = 0.0
    attempts: int = 1

    @property
    def truncated(self) -> bool:
        return self.finish_reason == "length"


@dataclass
class PageResult:
    """Final outcome for one page, after the optional second opinion."""

    text: str
    model: str
    confidence: PageConfidence | None  # of the primary transcription
    low_confidence: bool
    second_opinion_used: bool = False
    truncated: bool = False
    seconds: float = 0.0


def _error_message(resp: httpx.Response) -> str:
    try:
        body = resp.json()
    except ValueError:
        return resp.text[:300]
    if isinstance(body, dict):
        errs = body.get("errors")
        if isinstance(errs, list) and errs:
            return "; ".join(str(e.get("message", e)) if isinstance(e, dict) else str(e)
                             for e in errs)[:300]
        err = body.get("error")
        if isinstance(err, dict):
            return str(err.get("message") or err)[:300]
        if err:
            return str(err)[:300]
    return str(body)[:300]


def _retry_after(resp: httpx.Response) -> float | None:
    raw = resp.headers.get("retry-after")
    if not raw:
        return None
    try:
        return min(max(float(raw), 0.0), MAX_RETRY_AFTER)
    except ValueError:
        return None


class GatewayClient:
    def __init__(
        self,
        settings: Settings,
        http: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.s = settings
        self._own_http = http is None
        self.http = http or httpx.Client(
            timeout=httpx.Timeout(settings.timeout, connect=15.0),
            follow_redirects=False,
        )
        self._sleep = sleep

    # -- lifecycle -------------------------------------------------------------------------

    def close(self) -> None:
        if self._own_http:
            self.http.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- single request --------------------------------------------------------------------

    def _payload(self, model: str, image: bytes, mime: str, *, primary: bool,
                 logprobs: bool) -> dict[str, Any]:
        uri = f"data:{mime};base64,{base64.b64encode(image).decode('ascii')}"
        payload: dict[str, Any] = {
            "model": model,
            "max_tokens": self.s.max_tokens,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": self.s.prompt},
                    {"type": "image_url", "image_url": {"url": uri}},
                ],
            }],
        }
        if is_workers_ai(model):
            payload["temperature"] = 0
            if logprobs:
                payload["logprobs"] = True
        if primary:
            # User-supplied tuning targets the primary model only; e.g. Qwen-specific
            # parameters could be rejected by the second-opinion provider.
            payload.update(self.s.extra_params)
        return payload

    def transcribe(self, image: bytes, mime: str, model: str | None = None) -> Transcription:
        """Transcribe one page image with ``model`` (default: the primary model)."""
        model = model or self.s.model
        primary = model == self.s.model
        url = self.s.chat_completions_url(model)
        want_logprobs = is_workers_ai(model)
        t0 = time.monotonic()
        attempts = 0
        last_problem = "no attempt made"

        while attempts <= self.s.max_retries:
            attempts += 1
            payload = self._payload(model, image, mime, primary=primary, logprobs=want_logprobs)
            wait: float | None = None
            try:
                resp = self.http.post(url, json=payload, headers=self.s.headers())
            except (httpx.TimeoutException, httpx.TransportError) as e:
                last_problem = f"{type(e).__name__}: {e}"
            else:
                if resp.status_code == 200:
                    result = self._parse(resp, model, t0, attempts)
                    if result is not None:
                        return result
                    last_problem = "model returned an empty transcription"
                elif resp.status_code in (401, 403):
                    raise AuthError(
                        f"{model}: HTTP {resp.status_code} from AI Gateway ({_error_message(resp)}). "
                        "Check PAPERLESS_CF_OCR_API_TOKEN / PAPERLESS_CF_OCR_GATEWAY_TOKEN and "
                        "that the token may run Workers AI and use this gateway.",
                    )
                elif resp.status_code in RETRYABLE_STATUS:
                    last_problem = f"HTTP {resp.status_code}: {_error_message(resp)}"
                    wait = _retry_after(resp)
                else:
                    msg = _error_message(resp)
                    if want_logprobs and "logprob" in msg.lower():
                        logger.info("%s rejected logprobs; retrying without (no confidence score)",
                                    model)
                        want_logprobs = False
                        attempts -= 1
                        continue
                    raise OcrError(f"{model}: HTTP {resp.status_code}: {msg}")

            if attempts > self.s.max_retries:
                break
            if wait is None:
                wait = min(2 ** (attempts - 1) + random.uniform(0, 1), MAX_BACKOFF)
            logger.warning("%s attempt %d failed (%s); retrying in %.1fs",
                           model, attempts, last_problem, wait)
            self._sleep(wait)

        raise OcrError(f"{model}: giving up after {attempts} attempt(s): {last_problem}")

    def _parse(self, resp: httpx.Response, model: str, t0: float,
               attempts: int) -> Transcription | None:
        try:
            body = resp.json()
            choice = body["choices"][0]
            text = choice["message"].get("content") or ""
        except (ValueError, KeyError, IndexError, TypeError, AttributeError) as e:
            raise OcrError(f"{model}: unexpected response shape ({type(e).__name__})") from e
        finish = choice.get("finish_reason")
        if not text.strip():
            if finish == "length":
                raise OcrError(
                    f"{model}: output limit reached before any text "
                    f"(raise PAPERLESS_CF_OCR_MAX_TOKENS, currently {self.s.max_tokens})",
                )
            return None  # empty answer (e.g. reasoning consumed the turn): retry
        lp = choice.get("logprobs")
        tokens = lp.get("content") if isinstance(lp, dict) else None
        if finish == "length":
            logger.warning("%s: transcription truncated at max_tokens=%d", model, self.s.max_tokens)
        return Transcription(
            text=text.strip(),
            model=model,
            confidence=page_confidence(tokens, text),
            finish_reason=finish,
            usage=body.get("usage") or {},
            seconds=time.monotonic() - t0,
            attempts=attempts,
        )

    # -- page with optional second opinion -------------------------------------------------

    def ocr_page(self, image: bytes, mime: str, label: str = "page") -> PageResult:
        t0 = time.monotonic()
        second = self.s.second_opinion_model
        try:
            first = self.transcribe(image, mime)
        except AuthError:
            raise
        except OcrError as e:
            if not second:
                raise
            logger.warning("%s: primary model failed (%s); using second-opinion model %s",
                           label, e, second)
            alt = self.transcribe(image, mime, model=second)
            return PageResult(alt.text, alt.model, None, low_confidence=True,
                              second_opinion_used=True, truncated=alt.truncated,
                              seconds=time.monotonic() - t0)

        conf = first.confidence
        low = conf is not None and conf.is_low(self.s.low_confidence_ratio,
                                               self.s.low_confidence_min)
        if conf is not None:
            logger.info("%s: %s confidence %s%s", label, first.model, conf.describe(),
                        f" uncertain={list(conf.uncertain)}" if low else "")
        if low and second:
            try:
                alt = self.transcribe(image, mime, model=second)
            except OcrError as e:
                logger.warning("%s: second opinion with %s failed (%s); keeping %s output",
                               label, second, e, first.model)
            else:
                logger.info("%s: low confidence, using second opinion from %s", label, second)
                return PageResult(alt.text, alt.model, conf, low_confidence=True,
                                  second_opinion_used=True, truncated=alt.truncated,
                                  seconds=time.monotonic() - t0)
        return PageResult(first.text, first.model, conf, low_confidence=low,
                          truncated=first.truncated, seconds=time.monotonic() - t0)

    def ocr_pages(self, pages: Sequence[tuple[bytes, str]]) -> list[PageResult | OcrError]:
        """OCR pages concurrently. Each slot holds the result or the page's error, in order.

        An :class:`AuthError` is raised immediately: it affects every page.
        """
        out: list[PageResult | OcrError] = [OcrError("not processed")] * len(pages)
        with cf.ThreadPoolExecutor(max_workers=max(1, min(self.s.concurrency, len(pages)))) as ex:
            futs = {ex.submit(self.ocr_page, img, mime, f"page {i + 1}"): i
                    for i, (img, mime) in enumerate(pages)}
            for fut in cf.as_completed(futs):
                i = futs[fut]
                try:
                    out[i] = fut.result()
                except AuthError:
                    for f in futs:
                        f.cancel()
                    raise
                except OcrError as e:
                    out[i] = e
        return out
