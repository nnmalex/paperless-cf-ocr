#!/bin/bash
# Lab helper: borrow the `cf` CLI OAuth session (kept in memory only) and run eval.
# usage: tools/lab/run_eval.sh <include-glob>
set -euo pipefail
cd "$(dirname "$0")/../.."
. .venv/bin/activate
. tools/lab/cf_env.sh
python tools/lab/cfocr_lab.py eval tools/lab/samples --include "$1" ${EXTRA:-} --models ${MODELS:-mistral-small llama4-scout qwen3.8-27b gemma4-26b}
[ -n "${MODELS:-}" ] || python tools/lab/cfocr_lab.py eval tools/lab/samples --include "$1" --models moondream --variant direct
