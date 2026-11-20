#!/bin/sh
# Entry point of the installer image: replace the plugin in $TARGET with this image's copy.
# Runs on every `docker compose up`, so changing the image tag upgrades or downgrades.
set -eu
TARGET="${TARGET:-/out}"
if [ ! -d "$TARGET" ]; then
  echo "[paperless-cf-ocr] $TARGET is not mounted; nothing to do" >&2
  exit 1
fi
# Remove the previous version (including its *.dist-info, which carries the entry point).
find "$TARGET" -mindepth 1 -maxdepth 1 -exec rm -rf {} +
cp -a /plugin/. "$TARGET"/
# Readable by any container user (Paperless may run rootless), writable by nobody.
chmod -R a+rX,go-w "$TARGET"
version="$(ls "$TARGET" | sed -n 's/^paperless_cf_ocr-\(.*\)\.dist-info$/\1/p')"
echo "[paperless-cf-ocr] installed version ${version:-unknown} into $TARGET"
