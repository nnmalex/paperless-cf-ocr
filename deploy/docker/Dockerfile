# paperless-cf-ocr installer image.
#
# Not a Paperless image: it only carries the plugin, pre-installed into /plugin (pure
# Python, --no-deps: everything it imports ships with Paperless-ngx). Run it once before
# Paperless; it copies the plugin into a shared volume that Paperless adds to PYTHONPATH.
# See deploy/docker/compose.example.yml.
#
# Build (from the repo root):  docker build -f deploy/docker/Dockerfile -t paperless-cf-ocr .

FROM docker.io/library/python:3.14-slim AS build
WORKDIR /src
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir --root-user-action=ignore build \
 && python -m build --wheel --outdir /dist \
 && pip install --no-cache-dir --root-user-action=ignore --no-deps --no-compile \
      --target /plugin /dist/*.whl

FROM docker.io/library/busybox:1.37
LABEL org.opencontainers.image.title="paperless-cf-ocr installer" \
      org.opencontainers.image.description="Installs the paperless-cf-ocr parser plugin into a volume for Paperless-ngx" \
      org.opencontainers.image.licenses="GPL-3.0-only"
COPY --from=build /plugin /plugin
COPY deploy/docker/install.sh /install.sh
# Target directory inside the shared volume.
ENV TARGET=/out
ENTRYPOINT ["/bin/sh", "/install.sh"]
