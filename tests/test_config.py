from __future__ import annotations

from dataclasses import dataclass

import pytest

from paperless_cf_ocr import config as c

ACC = "0123456789abcdef0123456789abcdef"
ACC2 = "fedcba9876543210fedcba9876543210"
GW_URL = f"https://gateway.ai.cloudflare.com/v1/{ACC}/my-gw"


@dataclass
class AI:
    llm_backend: str | None = "openai-like"
    llm_endpoint: str | None = f"{GW_URL}/compat"
    llm_api_key: str | None = "cf-token-from-paperless"


def env(**kw: str) -> dict[str, str]:
    return {c.ENV_PREFIX + k: v for k, v in kw.items()}


# --- parse_gateway_url ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("url", "route"),
    [
        (GW_URL, None),
        (GW_URL + "/", None),
        (GW_URL + "/compat", "compat"),
        (GW_URL + "/workers-ai/v1", "workers-ai"),
        (GW_URL + "/openai", "openai"),
        ("  " + GW_URL + "/compat  ", "compat"),
    ],
)
def test_parse_gateway_url_valid(url: str, route: str | None) -> None:
    ref = c.parse_gateway_url(url)
    assert ref == c.GatewayRef(ACC, "my-gw", route)
    assert ref.base_url == GW_URL


@pytest.mark.parametrize(
    "url",
    [
        None,
        "",
        "https://api.openai.com/v1",
        f"http://gateway.ai.cloudflare.com/v1/{ACC}/my-gw",  # not https
        f"https://gateway.ai.cloudflare.com.evil.example/v1/{ACC}/my-gw",
        f"https://gateway.ai.cloudflare.com/v2/{ACC}/my-gw",
        "https://gateway.ai.cloudflare.com/v1/not-an-account/my-gw",
        f"https://gateway.ai.cloudflare.com/v1/{ACC}",
        f"https://gateway.ai.cloudflare.com/v1/{ACC}/bad%20id",
    ],
)
def test_parse_gateway_url_invalid(url: str | None) -> None:
    assert c.parse_gateway_url(url) is None


# --- plugin env only ------------------------------------------------------------------


def test_plugin_env_ids() -> None:
    r = c.resolve(env(ACCOUNT_ID=ACC, GATEWAY_ID="my-gw", API_TOKEN="t"))
    assert r.ok and not r.problems
    s = r.settings
    assert s.gateway.base_url == GW_URL
    assert s.chat_completions_url(s.model) == f"{GW_URL}/workers-ai/v1/chat/completions"
    assert s.chat_completions_url("anthropic/claude-sonnet-5-5") == f"{GW_URL}/compat/chat/completions"
    assert s.second_opinion_model is None
    assert s.model == c.DEFAULT_MODEL
    assert s.prompt == c.DEFAULT_PROMPT
    assert s.source == "plugin"
    assert s.headers()["Authorization"] == "Bearer t"
    assert "cf-aig-authorization" not in s.headers()
    assert s.headers()["User-Agent"].startswith("paperless-cf-ocr/")


def test_plugin_env_url_and_gateway_token() -> None:
    r = c.resolve(env(GATEWAY_URL=GW_URL + "/compat", API_TOKEN="t", GATEWAY_TOKEN="g"))
    h = r.settings.headers()
    assert h["Authorization"] == "Bearer t"
    assert h["cf-aig-authorization"] == "Bearer g"


def test_gateway_token_alone_is_enough() -> None:
    r = c.resolve(env(ACCOUNT_ID=ACC, GATEWAY_ID="my-gw", GATEWAY_TOKEN="g"))
    assert r.ok
    assert "Authorization" not in r.settings.headers()


def test_missing_credentials() -> None:
    r = c.resolve(env(ACCOUNT_ID=ACC, GATEWAY_ID="my-gw"))
    assert not r.ok
    assert "no credentials" in r.problems[0]


@pytest.mark.parametrize(
    ("e", "msg"),
    [
        (env(ACCOUNT_ID=ACC, API_TOKEN="t"), "set both"),
        (env(ACCOUNT_ID="xyz", GATEWAY_ID="g", API_TOKEN="t"), "32-character"),
        (env(ACCOUNT_ID=ACC, GATEWAY_ID="g/../x", API_TOKEN="t"), "invalid characters"),
        (env(GATEWAY_URL="https://example.com/v1", API_TOKEN="t"), "GATEWAY_URL"),
    ],
)
def test_bad_gateway_settings_are_reported_not_silently_replaced(e: dict, msg: str) -> None:
    # Even with valid Paperless AI settings available, an explicit but broken plugin
    # setting must surface as a problem rather than fall back quietly.
    r = c.resolve(e, AI())
    assert not r.ok
    assert any(msg in p for p in r.problems)


def test_nothing_configured() -> None:
    r = c.resolve({}, None)
    assert not r.ok
    assert "no Cloudflare AI Gateway configured" in r.problems[0]


# --- Paperless AI settings reuse ------------------------------------------------------


@pytest.mark.parametrize("route", ["compat", "workers-ai/v1"])
def test_reuse_paperless_ai_gateway_and_key(route: str) -> None:
    r = c.resolve({}, AI(llm_endpoint=f"{GW_URL}/{route}"))
    assert r.ok and not r.problems
    assert r.settings.gateway == c.GatewayRef(ACC, "my-gw")
    assert r.settings.api_token == "cf-token-from-paperless"
    assert r.settings.source == "paperless-ai"


def test_paperless_provider_route_key_not_reused() -> None:
    r = c.resolve({}, AI(llm_endpoint=f"{GW_URL}/openai", llm_api_key="sk-openai"))
    assert not r.ok
    assert any("'openai' provider route" in p for p in r.problems)
    # ...but supplying a token fixes it while still reusing the gateway location.
    r = c.resolve(env(API_TOKEN="t"), AI(llm_endpoint=f"{GW_URL}/openai", llm_api_key="sk"))
    assert r.ok and r.settings.api_token == "t" and r.settings.gateway.gateway_id == "my-gw"


def test_paperless_non_cloudflare_endpoint_ignored() -> None:
    r = c.resolve({}, AI(llm_endpoint="https://api.openai.com/v1"))
    assert not r.ok
    assert "no Cloudflare AI Gateway configured" in r.problems[0]


@pytest.mark.parametrize("backend", ["ollama", None, ""])
def test_paperless_non_openai_backend_ignored(backend: str | None) -> None:
    assert not c.resolve({}, AI(llm_backend=backend)).ok


def test_plugin_settings_override_paperless() -> None:
    r = c.resolve(
        env(ACCOUNT_ID=ACC2, GATEWAY_ID="other", API_TOKEN="mine", MODEL="@cf/x/y"),
        AI(),
    )
    s = r.settings
    assert (s.gateway.account_id, s.gateway.gateway_id) == (ACC2, "other")
    assert s.api_token == "mine" and s.model == "@cf/x/y" and s.source == "plugin"


def test_plugin_gateway_with_paperless_key_for_same_gateway() -> None:
    r = c.resolve(env(ACCOUNT_ID=ACC, GATEWAY_ID="my-gw"), AI())
    assert r.ok
    assert r.settings.api_token == "cf-token-from-paperless"
    assert r.settings.source == "plugin+paperless-ai-key"


def test_paperless_key_not_used_for_a_different_gateway() -> None:
    r = c.resolve(env(ACCOUNT_ID=ACC2, GATEWAY_ID="other"), AI())
    assert not r.ok
    assert any("no credentials" in p for p in r.problems)


# --- tuning, validation, disable ------------------------------------------------------


def base(**kw: str) -> dict[str, str]:
    return env(ACCOUNT_ID=ACC, GATEWAY_ID="my-gw", API_TOKEN="t", **kw)


def test_tuning_values() -> None:
    s = c.resolve(base(
        TIMEOUT="300", CONCURRENCY="8", DPI="300", MAX_IMAGE_PX="3000", MAX_PAGES="50",
        SCORE="25", ARCHIVE="NONE", FALLBACK="fail", MAX_RETRIES="0", EXISTING_TEXT="Trust",
        EXTRA_PARAMS='{"temperature": 0, "reasoning_effort": "low"}',
    )).settings
    assert (s.timeout, s.concurrency, s.dpi, s.max_image_px, s.max_pages, s.score) == (
        300.0, 8, 300, 3000, 50, 25)
    assert (s.archive, s.fallback, s.max_retries, s.existing_text) == ("none", "fail", 0, "trust")
    assert s.extra_params == {"temperature": 0, "reasoning_effort": "low"}


@pytest.mark.parametrize(
    ("kw", "attr", "default"),
    [
        ({"TIMEOUT": "abc"}, "timeout", 180.0),
        ({"TIMEOUT": "5"}, "timeout", 180.0),
        ({"CONCURRENCY": "0"}, "concurrency", 4),
        ({"ARCHIVE": "sometimes"}, "archive", "hybrid"),
        ({"EXTRA_PARAMS": "[1, 2]"}, "extra_params", {}),
        ({"EXTRA_PARAMS": "{nope"}, "extra_params", {}),
    ],
)
def test_invalid_values_fall_back_with_a_problem(kw: dict, attr: str, default: object) -> None:
    r = c.resolve(base(**kw))
    assert r.ok
    assert getattr(r.settings, attr) == default
    assert len(r.problems) == 1


def test_prompt_and_prompt_file(tmp_path) -> None:
    assert c.resolve(base(PROMPT="read it")).settings.prompt == "read it"
    f = tmp_path / "p.txt"
    f.write_text("from file\n")
    assert c.resolve(base(PROMPT_FILE=str(f))).settings.prompt == "from file"
    r = c.resolve(base(PROMPT_FILE=str(tmp_path / "missing.txt")))
    assert r.settings.prompt == c.DEFAULT_PROMPT and "PROMPT_FILE" in r.problems[0]


def test_blank_values_are_treated_as_unset() -> None:
    r = c.resolve(base(MODEL="   ", GATEWAY_TOKEN=""))
    assert r.settings.model == c.DEFAULT_MODEL and r.settings.gateway_token is None


@pytest.mark.parametrize("v", ["false", "0", "no", "OFF"])
def test_disabled(v: str) -> None:
    r = c.resolve(base(ENABLED=v), AI())
    assert not r.enabled and not r.ok and r.settings is None


def test_describe_never_contains_secrets() -> None:
    s = c.resolve(base(GATEWAY_TOKEN="super-secret-g"), AI()).settings
    d = s.describe()
    assert "t" not in d.split("auth=")[1].replace("api-token", "").replace("gateway-token", "")
    assert "super-secret" not in d and ACC not in d


def test_paperless_ai_settings_outside_paperless_returns_none() -> None:
    assert c.paperless_ai_settings() is None


def test_second_opinion_and_confidence_settings() -> None:
    s = c.resolve(base(SECOND_OPINION_MODEL="anthropic/claude-sonnet-5-5",
                       LOW_CONFIDENCE_RATIO="0.02", LOW_CONFIDENCE_MIN="0.1",
                       MAX_TOKENS="4096")).settings
    assert s.second_opinion_model == "anthropic/claude-sonnet-5-5"
    assert (s.low_confidence_ratio, s.low_confidence_min, s.max_tokens) == (0.02, 0.1, 4096)
    assert "second_opinion=anthropic/claude-sonnet-5-5" in s.describe()


@pytest.mark.parametrize(("kw", "attr", "default"), [
    ({"LOW_CONFIDENCE_RATIO": "lots"}, "low_confidence_ratio", 0.01),
    ({"LOW_CONFIDENCE_MIN": "1.5"}, "low_confidence_min", 0.2),
])
def test_invalid_floats(kw: dict, attr: str, default: float) -> None:
    r = c.resolve(base(**kw))
    assert getattr(r.settings, attr) == default and len(r.problems) == 1


def test_second_opinion_mode() -> None:
    assert c.resolve(base()).settings.second_opinion == "low-confidence"
    s = c.resolve(base(SECOND_OPINION="Always", SECOND_OPINION_MODEL="anthropic/x")).settings
    assert s.second_opinion == "always" and "second_opinion=anthropic/x(always)" in s.describe()
    r = c.resolve(base(SECOND_OPINION="sometimes"))
    assert r.settings.second_opinion == "low-confidence" and len(r.problems) == 1
