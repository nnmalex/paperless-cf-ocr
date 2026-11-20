#!/bin/bash
# Download the Paperless-ngx modules the integration tests run for real (not vendored:
# Paperless-ngx is GPL-3.0; these files stay out of the repo, see .gitignore).
# usage: tests_integration/fetch_paperless.sh [tag]   (default v3.5.1)
set -euo pipefail
TAG="${1:-v3.5.1}"
DEST="$(cd "$(dirname "$0")" && pwd)/_paperless"
BASE="https://raw.githubusercontent.com/paperless-ngx/paperless-ngx/${TAG}/src"
for f in paperless/parsers/tesseract.py paperless/parsers/utils.py documents/parsers.py documents/utils.py; do
  mkdir -p "$DEST/$(dirname "$f")"
  curl -fsSL -A paperless-cf-ocr-tests "$BASE/$f" -o "$DEST/$f"
done
echo "$TAG" > "$DEST/TAG"
echo "Fetched Paperless-ngx $TAG modules into $DEST"
