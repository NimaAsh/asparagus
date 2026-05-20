#!/usr/bin/env bash
#
# Install `mosaicml-streaming` into an existing asparagus uv environment
# without dragging in the full transformers/huggingface-hub dependency tree
# that triggers a conflict against asparagus' own pins. Our usage is just
# `streaming.StreamingDataset` for local MDS shards, which only needs the
# core package plus `xxhash` (hash validation) and `zstandard` (compressed
# shard reads).
#
# Run this AFTER `uv sync --group dcai --extra extras`.
#
# Usage:
#   bash scripts/install_streaming.sh
#
# Honors UV_PROJECT_ENVIRONMENT, falling back to the asparagus default of
# /data/$USER/asparagus/venvs/asparagus.
set -euo pipefail

if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: uv not on PATH. Source your env or 'export PATH=\"\$HOME/.local/bin:\$PATH\"' first." >&2
  exit 127
fi

VENV="${UV_PROJECT_ENVIRONMENT:-/data/$USER/asparagus/venvs/asparagus}"
export UV_PROJECT_ENVIRONMENT="$VENV"

# `--no-deps` is the key bit: skip transformers / huggingface-hub / numpy
# downgrades that mosaicml-streaming would otherwise demand.
uv pip install --no-deps mosaicml-streaming
uv pip install --no-deps xxhash zstandard

echo "==> Smoke test"
uv run python - <<'PY'
from streaming import StreamingDataset
print("OK: streaming.StreamingDataset is importable")
PY
