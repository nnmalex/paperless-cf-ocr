"""Page confidence from token log-probabilities.

Validated in tools/lab (milestone 1b): on clean pages ~0% of words fall below p=0.5 and
none below 0.2, while degraded scans and handwriting show 2-3% low words, and the
lowest-probability words are the actual misreads.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

ILLEGIBLE = "[illegible]"
_REASONING_END = "</think>"
_SPECIAL = ("<|im_end|>", "<|endoftext|>", "</s>")


@dataclass(frozen=True)
class PageConfidence:
    mean: float  # mean word probability
    low_ratio: float  # share of words with probability < 0.5
    min: float  # lowest word probability
    illegible: int  # number of [illegible] markers in the text
    words: int
    uncertain: tuple[str, ...] = field(default=())  # lowest-probability words, for logs

    def is_low(self, max_low_ratio: float, min_word_prob: float) -> bool:
        return self.low_ratio > max_low_ratio or self.min < min_word_prob or self.illegible > 0

    def describe(self) -> str:
        return (f"mean={self.mean:.3f} low={self.low_ratio:.1%} min={self.min:.2f} "
                f"illegible={self.illegible}")


def word_probabilities(tokens: Iterable[Mapping[str, Any]]) -> list[tuple[str, float]]:
    """Group tokens into whitespace-separated words; a word's probability is its weakest token.

    Tokens up to and including a reasoning terminator (``</think>``) are dropped: some
    models (Qwen) return log-probabilities for their hidden reasoning too.
    """
    words: list[tuple[str, float]] = []
    cur, cur_p = "", 1.0
    for t in tokens:
        tok = str(t.get("token", ""))
        try:
            p = math.exp(float(t["logprob"]))
        except (KeyError, TypeError, ValueError, OverflowError):
            continue
        if tok[:1].isspace():
            if cur.strip():
                words.append((cur.strip(), cur_p))
            cur, cur_p = tok, p
        else:
            cur, cur_p = cur + tok, min(cur_p, p)
    if cur.strip():
        words.append((cur.strip(), cur_p))

    cut = max((i for i, (w, _) in enumerate(words) if _REASONING_END in w), default=-1)
    out = []
    for w, p in words[cut + 1:]:
        for s in _SPECIAL:
            w = w.replace(s, "")
        if w:
            out.append((w, p))
    return out


def page_confidence(tokens: Iterable[Mapping[str, Any]] | None, text: str) -> PageConfidence | None:
    """Score a page, or None when the response carried no usable log-probabilities."""
    if not tokens:
        return None
    words = word_probabilities(tokens)
    if not words:
        return None
    probs = [p for _, p in words]
    low = sum(1 for p in probs if p < 0.5)
    uncertain = tuple(w for w, p in sorted(words, key=lambda x: x[1])[:10] if p < 0.6)
    return PageConfidence(
        mean=sum(probs) / len(probs),
        low_ratio=low / len(probs),
        min=min(probs),
        illegible=text.lower().count(ILLEGIBLE),
        words=len(words),
        uncertain=uncertain,
    )
