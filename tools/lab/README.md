# Milestone 1 lab

Throwaway tooling to (1) verify how to call Workers AI through AI Gateway and
(2) pick the default OCR model. Not part of the plugin package.

```sh
python3 -m venv .venv && . .venv/bin/activate
pip install -r tools/lab/requirements.txt
cp tools/lab/.env.example tools/lab/.env   # fill in

python tools/lab/cfocr_lab.py probe                 # which URL / header / model combos work
python tools/lab/cfocr_lab.py eval tools/lab/samples  # accuracy, latency, tokens per model
```

`probe` checks, per model: the native Workers AI route
`{gw}/workers-ai/{model}`, the OpenAI-style route `{gw}/workers-ai/v1/chat/completions`,
and the unified `{gw}/compat/chat/completions` route; and, if `CF_AIG_TOKEN` is set,
that the gateway rejects requests without `cf-aig-authorization`.

`eval` writes per-page transcriptions, `pages.csv` and `summary.md` to `results/<timestamp>/`.
Score: CER/WER after normalizing case, whitespace and markdown/table punctuation.
