from __future__ import annotations

import math

import pytest

from paperless_cf_ocr.confidence import page_confidence, word_probabilities


def toks(*pairs: tuple[str, float]) -> list[dict]:
    return [{"token": t, "logprob": math.log(p)} for t, p in pairs]


def test_words_take_weakest_token() -> None:
    words = word_probabilities(toks(("Hart", 0.9), ("we", 0.4), ("ll", 0.95), (" House", 0.99)))
    assert words == [("Hartwell", pytest.approx(0.4)), ("House", pytest.approx(0.99))]


def test_newline_tokens_split_words() -> None:
    words = word_probabilities(toks(("Line", 0.9), ("\n", 0.9), ("two", 0.8)))
    assert [w for w, _ in words] == ["Line", "two"]


def test_reasoning_and_special_tokens_dropped() -> None:
    words = word_probabilities(toks(("hmm", 0.1), (" maybe", 0.1), ("</think>", 1.0),
                                    ("\nText", 0.99), ("<|im_end|>", 0.99)))
    assert words == [("Text", pytest.approx(0.99))]


def test_page_confidence_values() -> None:
    conf = page_confidence(toks(("a", 0.99), (" b", 0.4), (" c", 0.15), (" d", 0.95)), "a b c d")
    assert conf.words == 4
    assert conf.low_ratio == pytest.approx(0.5)
    assert conf.min == pytest.approx(0.15)
    assert conf.uncertain == ("c", "b")
    assert conf.is_low(0.01, 0.2)
    assert not page_confidence(toks(("a", 0.99), (" b", 0.98)), "a b").is_low(0.01, 0.2)


def test_illegible_marker_makes_page_low() -> None:
    conf = page_confidence(toks(("x", 0.99)), "x [illegible] y [ILLEGIBLE]")
    assert conf.illegible == 2 and conf.is_low(0.5, 0.0)


@pytest.mark.parametrize("tokens", [None, [], [{"token": "a"}], [{"token": "a", "logprob": "x"}]])
def test_unusable_logprobs(tokens) -> None:
    assert page_confidence(tokens, "a") is None
