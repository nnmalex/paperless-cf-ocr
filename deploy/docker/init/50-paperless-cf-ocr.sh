#!/bin/bash
# Alternative Docker install: pip-install paperless-cf-ocr from PyPI at container start.
#
# Mount the directory containing this script at /custom-cont-init.d (read-only, owned by
# root, mode 755; the script itself 744). Paperless-ngx runs it as root before starting.
# Needs internet access on every start and does not work for rootless containers; prefer
# the installer image (compose.example.yml) where possible.
#
# Optional: PAPERLESS_CF_OCR_VERSION=0.1.0 pins the version (recommended);
#           PAPERLESS_CF_OCR_INSTALL_SOURCE=<path or pip spec> installs from elsewhere.
# A failure never blocks Paperless: it starts with its built-in parsers instead.

spec="paperless-cf-ocr"
[ -n "${PAPERLESS_CF_OCR_VERSION:-}" ] && spec="paperless-cf-ocr==${PAPERLESS_CF_OCR_VERSION}"
# Or install from somewhere else, e.g. a mounted wheel: /wheels/paperless_cf_ocr-0.1.0-py3-none-any.whl
[ -n "${PAPERLESS_CF_OCR_INSTALL_SOURCE:-}" ] && spec="${PAPERLESS_CF_OCR_INSTALL_SOURCE}"

echo "[paperless-cf-ocr] installing ${spec}"
# --no-deps: everything the plugin imports ships with Paperless; never touch its pins.
if command -v uv >/dev/null 2>&1; then
  installer=(uv pip install --system --no-deps --quiet --reinstall)
else
  installer=(python3 -m pip install --no-deps --quiet --force-reinstall --root-user-action=ignore)
fi
if ! "${installer[@]}" "$spec"; then
  echo "[paperless-cf-ocr] WARNING: installation failed; Paperless will run without the plugin"
  exit 0
fi

if python3 -c "import importlib.metadata as m; \
eps = m.entry_points(group='paperless_ngx.parsers'); \
assert any(e.name == 'cf_ocr' for e in eps)"; then
  echo "[paperless-cf-ocr] installed $(python3 -c 'import importlib.metadata as m; print(m.version("paperless-cf-ocr"))')"
else
  echo "[paperless-cf-ocr] WARNING: package installed but its entry point is not visible"
fi
exit 0
