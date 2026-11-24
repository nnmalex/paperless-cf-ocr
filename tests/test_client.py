from __future__ import annotations

import json
import math

import httpx
import pytest
import respx

from paperless_cf_ocr import client as cl
from paperless_cf_ocr import config as c

ACC = "0123456789abcdef0123456789abcdef"
BASE = f"https://gateway.ai.cloudflare.com/v1/{ACC}/gw"
WAI = f"{BASE}/workers-ai/v1/chat/completions"
COMPAT = f"{BASE}/compat/chat/completions"
SONNET = "anthropic/claude-sonnet-5-5"


def settings(**kw: str) -> c.Settings:
    env = {"ACCOUNT_ID": ACC, "GATEWAY_ID": "gw", "API_TOKEN": "tok", "GATEWAY_TOKEN": "gtok"}
    env.update(kw)
    r = c.resolve({c.ENV_PREFIX + k: v for k, v in env.items()})
    assert r.ok, r.problems
    return r.settings


def lp(words: list[tuple[str, float]], reasoning: str | None = None) -> dict:
    toks = []
    if reasoning:
        toks += [{"token": w, "logprob": math.log(0.3)} for w in reasoning.split()]
        toks.append({"token": "</think>", "logprob": 0.0})
    for i, (w, p) in enumerate(words):
        toks.append({"token": (" " if i else "\n") + w, "logprob": math.log(p)})
    return {"content": toks}


def ok(text: str, logprobs: dict | None = None, finish: str = "stop") -> httpx.Response:
    choice = {"index": 0, "message": {"role": "assistant", "content": text},
              "finish_reason": finish}
    if logprobs is not None:
        choice["logprobs"] = logprobs
    return httpx.Response(200, json={"choices": [choice], "usage": {"prompt_tokens": 10}})


CLEAN = lp([("Hello", 0.99), ("world", 0.98)])
SHAKY = lp([("Dr", 0.95), ("Smiht", 0.1), ("House", 0.9)])


@pytest.fixture
def gw():
    with respx.mock(assert_all_called=False) as m:
        yield m


def mk(s: c.Settings | None = None) -> tuple[cl.GatewayClient, list[float]]:
    sleeps: list[float] = []
    return cl.GatewayClient(s or settings(), sleep=sleeps.append), sleeps


# --- request shape --------------------------------------------------------------------


def test_workers_ai_request_shape(gw) -> None:
    route = gw.post(WAI).mock(return_value=ok("Hello world", CLEAN))
    client, _ = mk(settings(EXTRA_PARAMS='{"reasoning_effort": "low"}'))
    t = client.transcribe(b"\x89PNG", "image/png")
    req = route.calls.last.request
    body = json.loads(req.content)
    assert body["model"] == c.DEFAULT_MODEL
    assert body["temperature"] == 0 and body["logprobs"] is True
    assert body["reasoning_effort"] == "low"
    assert body["max_tokens"] == 8192
    content = body["messages"][0]["content"]
    assert content[0] == {"type": "text", "text": c.DEFAULT_PROMPT}
    assert content[1]["image_url"]["url"] == "data:image/png;base64,iVBORw=="
    assert req.headers["authorization"] == "Bearer tok"
    assert req.headers["cf-aig-authorization"] == "Bearer gtok"
    assert req.headers["user-agent"].startswith("paperless-cf-ocr/")
    assert t.text == "Hello world" and t.confidence is not None and t.attempts == 1


def test_default_model_gets_reasoning_disabled(gw) -> None:
    route = gw.post(WAI).mock(return_value=ok("Hello world", CLEAN))
    client, _ = mk()
    client.transcribe(b"x", "image/png")
    body = json.loads(route.calls.last.request.content)
    assert body["model"] == c.DEFAULT_MODEL and body["reasoning_effort"] == "none"


def test_other_models_get_no_model_defaults(gw) -> None:
    route = gw.post(WAI).mock(return_value=ok("Hello world", CLEAN))
    client, _ = mk(settings(MODEL="@cf/qwen/qwen3.8-27b"))
    client.transcribe(b"x", "image/png")
    assert "reasoning_effort" not in json.loads(route.calls.last.request.content)


def test_model_defaults_apply_to_second_opinion_model(gw) -> None:
    route = gw.post(WAI).mock(return_value=ok("Hello world", CLEAN))
    client, _ = mk(settings(MODEL=SONNET, SECOND_OPINION_MODEL=c.DEFAULT_MODEL,
                            SECOND_OPINION="always"))
    client.ocr_page(b"x", "image/png")
    assert json.loads(route.calls.last.request.content)["reasoning_effort"] == "none"


def test_third_party_model_uses_compat_without_sampling_or_extra_params(gw) -> None:
    route = gw.post(COMPAT).mock(return_value=ok("Hi"))
    client, _ = mk(settings(EXTRA_PARAMS='{"reasoning_effort": "low"}'))
    t = client.transcribe(b"x", "image/jpeg", model=SONNET)
    body = json.loads(route.calls.last.request.content)
    assert body["model"] == SONNET
    assert not {"temperature", "logprobs", "reasoning_effort"} & body.keys()
    assert t.confidence is None


def test_third_party_primary_gets_extra_params(gw) -> None:
    route = gw.post(COMPAT).mock(return_value=ok("Hi"))
    client, _ = mk(settings(MODEL=SONNET, EXTRA_PARAMS='{"top_k": 5}'))
    client.transcribe(b"x", "image/jpeg")
    assert json.loads(route.calls.last.request.content)["top_k"] == 5


# --- retries and errors ---------------------------------------------------------------


@pytest.mark.parametrize("status", [408, 429, 500, 502, 503, 504, 524])
def test_retryable_status_then_success(gw, status: int) -> None:
    gw.post(WAI).mock(side_effect=[httpx.Response(status, json={"errors": [{"message": "x"}]}),
                                   ok("done", CLEAN)])
    client, sleeps = mk()
    t = client.transcribe(b"x", "image/png")
    assert t.text == "done" and t.attempts == 2 and len(sleeps) == 1


def test_retry_after_header_is_honoured_and_capped(gw) -> None:
    gw.post(WAI).mock(side_effect=[httpx.Response(429, headers={"retry-after": "7"}),
                                   httpx.Response(429, headers={"retry-after": "999"}),
                                   ok("done", CLEAN)])
    client, sleeps = mk()
    client.transcribe(b"x", "image/png")
    assert sleeps == [7.0, cl.MAX_RETRY_AFTER]


def test_transport_errors_are_retried(gw) -> None:
    gw.post(WAI).mock(side_effect=[httpx.ReadTimeout("slow"), httpx.ConnectError("down"),
                                   ok("done", CLEAN)])
    client, sleeps = mk()
    assert client.transcribe(b"x", "image/png").attempts == 3
    assert len(sleeps) == 2 and all(s <= cl.MAX_BACKOFF for s in sleeps)


def test_gives_up_after_max_retries(gw) -> None:
    route = gw.post(WAI).mock(return_value=httpx.Response(503, text="busy"))
    client, sleeps = mk(settings(MAX_RETRIES="2"))
    with pytest.raises(cl.OcrError, match="giving up after 3 attempt"):
        client.transcribe(b"x", "image/png")
    assert route.call_count == 3 and len(sleeps) == 2


@pytest.mark.parametrize("status", [401, 403])
def test_auth_errors_fail_fast(gw, status: int) -> None:
    route = gw.post(WAI).mock(return_value=httpx.Response(
        status, json={"success": False, "errors": [{"code": 2009, "message": "Unauthorized"}]}))
    client, sleeps = mk(settings(API_TOKEN="s3cr3t-api", GATEWAY_TOKEN="s3cr3t-gw"))
    with pytest.raises(cl.AuthError, match="Unauthorized") as ei:
        client.transcribe(b"x", "image/png")
    assert route.call_count == 1 and not sleeps
    assert "s3cr3t" not in str(ei.value)  # never echo secrets


def test_bad_request_is_not_retried(gw) -> None:
    route = gw.post(WAI).mock(return_value=httpx.Response(
        400, json={"error": {"message": "image too large"}}))
    client, _ = mk()
    with pytest.raises(cl.OcrError, match="HTTP 400: image too large"):
        client.transcribe(b"x", "image/png")
    assert route.call_count == 1


def test_logprobs_rejection_retries_without_them(gw) -> None:
    route = gw.post(WAI).mock(side_effect=[
        httpx.Response(400, json={"errors": [{"message": "logprobs is not supported"}]}),
        ok("plain"),
    ])
    client, sleeps = mk()
    t = client.transcribe(b"x", "image/png")
    assert t.text == "plain" and t.confidence is None and not sleeps
    assert "logprobs" not in json.loads(route.calls.last.request.content)


def test_empty_answer_is_retried(gw) -> None:
    gw.post(WAI).mock(side_effect=[ok("   "), ok("now text", CLEAN)])
    client, sleeps = mk()
    assert client.transcribe(b"x", "image/png").text == "now text"
    assert len(sleeps) == 1


def test_empty_answer_at_length_limit_is_an_error(gw) -> None:
    gw.post(WAI).mock(return_value=ok("", finish="length"))
    client, _ = mk()
    with pytest.raises(cl.OcrError, match="MAX_TOKENS"):
        client.transcribe(b"x", "image/png")


def test_truncated_answer_is_kept_and_flagged(gw) -> None:
    gw.post(WAI).mock(return_value=ok("partial text", CLEAN, finish="length"))
    client, _ = mk()
    assert client.transcribe(b"x", "image/png").truncated


LOOP = "Name: Dr X\n" + "[illegible]\n" * 200


def test_repetition_loop_is_retried(gw) -> None:
    gw.post(WAI).mock(side_effect=[ok(LOOP, CLEAN, finish="length"), ok("now text", CLEAN)])
    client, sleeps = mk()
    assert client.transcribe(b"x", "image/png").text == "now text"
    assert len(sleeps) == 1


def test_persistent_repetition_loop_is_an_error(gw) -> None:
    gw.post(WAI).mock(return_value=ok(LOOP, CLEAN, finish="length"))
    client, _ = mk(settings(MAX_RETRIES="1"))
    with pytest.raises(cl.OcrError, match="repeating itself"):
        client.transcribe(b"x", "image/png")


@pytest.mark.parametrize(("text", "looping"), [
    ("[illegible]\n" * 100, True),
    ("ab" * 400, True),
    ("x" * 1000, True),
    ("short\n" * 10, False),  # below the window
    ("".join(f"row {i} | value {i * 7}\n" for i in range(100)), False),  # long table
    (("[illegible]\n" * 100) + "Signed: Dr X", False),  # loop not at the end
])
def test_is_looping(text: str, looping: bool) -> None:
    assert cl.is_looping(text) is looping


def test_malformed_response(gw) -> None:
    gw.post(WAI).mock(return_value=httpx.Response(200, json={"result": {}}))
    client, _ = mk()
    with pytest.raises(cl.OcrError, match="unexpected response shape"):
        client.transcribe(b"x", "image/png")


# --- confidence and second opinion ----------------------------------------------------


def test_reasoning_tokens_excluded_from_confidence(gw) -> None:
    gw.post(WAI).mock(return_value=ok("Hello world", lp([("Hello", 0.99), ("world", 0.99)],
                                                        reasoning="let me think hmm")))
    client, _ = mk()
    conf = client.transcribe(b"x", "image/png").confidence
    assert conf.words == 2 and conf.min > 0.9


def test_confident_page_does_not_escalate(gw) -> None:
    gw.post(WAI).mock(return_value=ok("Hello world", CLEAN))
    compat = gw.post(COMPAT).mock(return_value=ok("sonnet"))
    client, _ = mk(settings(SECOND_OPINION_MODEL=SONNET))
    r = client.ocr_page(b"x", "image/png")
    assert r.text == "Hello world" and not r.low_confidence and not r.second_opinion_used
    assert compat.call_count == 0


def test_low_confidence_page_uses_second_opinion_output(gw) -> None:
    gw.post(WAI).mock(return_value=ok("Dr Smiht House", SHAKY))
    gw.post(COMPAT).mock(return_value=ok("DR J SMITH"))
    client, _ = mk(settings(SECOND_OPINION_MODEL=SONNET))
    r = client.ocr_page(b"x", "image/png")
    assert r.text == "DR J SMITH" and r.model == SONNET
    assert r.low_confidence and r.second_opinion_used
    assert r.confidence.min == pytest.approx(0.1)  # primary's score is kept for reporting


def test_illegible_marker_triggers_second_opinion(gw) -> None:
    gw.post(WAI).mock(return_value=ok("Dose [illegible] mg", lp([("Dose", .99), ("[illegible]", .99),
                                                                ("mg", .99)])))
    gw.post(COMPAT).mock(return_value=ok("Dose 0.3 mg"))
    client, _ = mk(settings(SECOND_OPINION_MODEL=SONNET))
    assert client.ocr_page(b"x", "image/png").text == "Dose 0.3 mg"


def test_low_confidence_without_second_opinion_keeps_primary(gw) -> None:
    gw.post(WAI).mock(return_value=ok("Dr Smiht House", SHAKY))
    client, _ = mk()
    r = client.ocr_page(b"x", "image/png")
    assert r.text == "Dr Smiht House" and r.low_confidence and not r.second_opinion_used


def test_second_opinion_failure_keeps_primary(gw) -> None:
    gw.post(WAI).mock(return_value=ok("Dr Smiht House", SHAKY))
    gw.post(COMPAT).mock(return_value=httpx.Response(400, json={"error": {"message": "nope"}}))
    client, _ = mk(settings(SECOND_OPINION_MODEL=SONNET))
    r = client.ocr_page(b"x", "image/png")
    assert r.text == "Dr Smiht House" and not r.second_opinion_used


def test_primary_failure_falls_back_to_second_opinion_model(gw) -> None:
    gw.post(WAI).mock(return_value=httpx.Response(400, json={"error": {"message": "bad"}}))
    gw.post(COMPAT).mock(return_value=ok("rescued"))
    client, _ = mk(settings(SECOND_OPINION_MODEL=SONNET))
    r = client.ocr_page(b"x", "image/png")
    assert r.text == "rescued" and r.second_opinion_used


def test_no_logprobs_means_no_escalation(gw) -> None:
    gw.post(COMPAT).mock(return_value=ok("text"))
    client, _ = mk(settings(MODEL=SONNET, SECOND_OPINION_MODEL="@cf/qwen/qwen3.8-27b"))
    r = client.ocr_page(b"x", "image/png")
    assert r.confidence is None and not r.low_confidence and not r.second_opinion_used


def test_low_confidence_thresholds_configurable(gw) -> None:
    gw.post(WAI).mock(return_value=ok("Dr Smiht House", SHAKY))
    compat = gw.post(COMPAT).mock(return_value=ok("x"))
    client, _ = mk(settings(SECOND_OPINION_MODEL=SONNET, LOW_CONFIDENCE_RATIO="0.5",
                            LOW_CONFIDENCE_MIN="0.05"))
    assert not client.ocr_page(b"x", "image/png").low_confidence
    assert compat.call_count == 0


# --- many pages -----------------------------------------------------------------------


def test_ocr_pages_keeps_order_and_isolates_failures(gw) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        url = json.loads(request.content)["messages"][0]["content"][1]["image_url"]["url"]
        if url.endswith("Mg=="):  # page b"2"
            return httpx.Response(400, json={"error": {"message": "bad page"}})
        return ok(f"text {url[-4:]}", CLEAN)

    gw.post(WAI).mock(side_effect=handler)
    client, _ = mk(settings(CONCURRENCY="3"))
    out = client.ocr_pages([(b"1", "image/png"), (b"2", "image/png"), (b"3", "image/png")])
    assert isinstance(out[0], cl.PageResult) and out[0].text == "text MQ=="
    assert isinstance(out[1], cl.OcrError) and "bad page" in str(out[1])
    assert isinstance(out[2], cl.PageResult) and out[2].text == "text Mw=="


def test_ocr_pages_auth_error_aborts(gw) -> None:
    gw.post(WAI).mock(return_value=httpx.Response(401, json={"errors": [{"message": "no"}]}))
    client, _ = mk()
    with pytest.raises(cl.AuthError):
        client.ocr_pages([(b"1", "image/png"), (b"2", "image/png")])


def test_always_mode_uses_second_model_only(gw) -> None:
    wai = gw.post(WAI).mock(return_value=ok("qwen", CLEAN))
    gw.post(COMPAT).mock(return_value=ok("sonnet"))
    client, _ = mk(settings(SECOND_OPINION_MODEL=SONNET, SECOND_OPINION="always"))
    r = client.ocr_page(b"x", "image/png")
    assert r.text == "sonnet" and r.model == SONNET and r.second_opinion_used
    assert wai.call_count == 0


def test_always_mode_falls_back_to_primary(gw) -> None:
    gw.post(WAI).mock(return_value=ok("Dr Smiht House", SHAKY))
    compat = gw.post(COMPAT).mock(return_value=httpx.Response(400, json={"error": {"message": "x"}}))
    client, _ = mk(settings(SECOND_OPINION_MODEL=SONNET, SECOND_OPINION="always"))
    r = client.ocr_page(b"x", "image/png")
    assert r.text == "Dr Smiht House" and not r.second_opinion_used
    assert r.low_confidence and compat.call_count == 1  # no second call to the failed model


def test_always_without_model_is_primary_only(gw) -> None:
    gw.post(WAI).mock(return_value=ok("qwen", CLEAN))
    client, _ = mk(settings(SECOND_OPINION="always"))
    assert client.ocr_page(b"x", "image/png").text == "qwen"
