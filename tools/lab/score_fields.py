#!/usr/bin/env python3
"""Key-field checklist scoring for lab results.

samples/fields.json (git-ignored; contains personal data):
  {doc_stem: {page: {"must": {label: [alternatives...]}, "must_not": {label: [...]}}}}
Matching is case/whitespace/quote-insensitive substring search on the page text.
For "must" fields that are space-separated value groups (e.g. "-1.75 -1.25 170"), table
separators like '|' are ignored by normalization, so row-wise layouts still match.

usage: score_fields.py RESULTS_DIR [RESULTS_DIR ...]   (also scores results/ref text layers)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from cfocr_lab import normalize  # noqa: E402


def norm(s: str) -> str:
    return normalize(s).replace(" ,", ",")


def main() -> None:
    fields = json.loads((HERE / "samples" / "fields.json").read_text())
    sources: dict[str, Path] = {"existing text layer": HERE / "results" / "ref"}
    for d in sys.argv[1:]:
        for m in sorted(Path(d).iterdir()):
            if m.is_dir():
                sources[m.name] = m
    for doc, pages in fields.items():
        print(f"\n### {doc}")
        labels = [(p, "must", k) for p, f in pages.items() for k in f["must"]] + \
                 [(p, "must_not", k) for p, f in pages.items() for k in f["must_not"]]
        table = {}
        for name, d in sources.items():
            row = {}
            for p, kind, k in labels:
                fp = d / (f"{doc}.p{p}.textlayer.txt" if name == "existing text layer" else f"{doc}.p{p}.txt")
                if not fp.exists():
                    row[(p, kind, k)] = None
                    continue
                txt = norm(fp.read_text())
                hit = any(norm(alt) in txt for alt in pages[p][kind][k])
                row[(p, kind, k)] = hit if kind == "must" else not hit
            table[name] = row
        names = [n for n in sources if any(v is not None for v in table[n].values())]
        print("| field | " + " | ".join(names) + " |")
        print("|---|" + "---|" * len(names))
        for key in labels:
            p, kind, k = key
            cells = ["–" if table[n][key] is None else ("✓" if table[n][key] else "✗") for n in names]
            print(f"| p{p} {'NOT ' if kind == 'must_not' else ''}{k} | " + " | ".join(cells) + " |")
        tot = [sum(1 for v in table[n].values() if v) for n in names]
        print("| **score** | " + " | ".join(f"**{t}/{len(labels)}**" for t in tot) + " |")


if __name__ == "__main__":
    main()
