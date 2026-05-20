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
VENV_PY="$VENV/bin/python"

if [[ ! -x "$VENV_PY" ]]; then
  echo "ERROR: $VENV_PY does not exist or is not executable." >&2
  echo "Make sure you ran 'uv sync --group dcai --extra extras' first." >&2
  exit 1
fi

echo "==> Target python: $VENV_PY"
"$VENV_PY" -c "import sys, sysconfig; print('exe:', sys.executable); print('site:', sysconfig.get_paths()['purelib'])"

# `--no-deps` is the key bit: skip transformers / huggingface-hub / numpy
# downgrades that mosaicml-streaming would otherwise demand.
# `--python "$VENV_PY"` pins the install to *that* env and bypasses uv's
# auto-discovery (which prefers a local `.venv` or another project env).
uv pip install --python "$VENV_PY" --no-deps mosaicml-streaming
uv pip install --python "$VENV_PY" --no-deps xxhash zstandard

echo "==> Smoke test"
"$VENV_PY" - <<'PY'
from streaming import StreamingDataset
import streaming
print("OK: streaming.StreamingDataset is importable")
print("streaming.__file__:", streaming.__file__)
PY
