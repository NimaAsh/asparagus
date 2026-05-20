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

# `streaming.base.dataloader` does an eager top-level
# `from transformers.feature_extraction_utils import BatchFeature`, so the
# `transformers` namespace has to exist with that one symbol. Installing
# real transformers with --no-deps fails because transformers internals
# eagerly import `regex`, then `tokenizers`, then `safetensors`, etc.
# Installing with deps re-introduces the huggingface-hub<1.0 / numpy<2.0
# conflicts. Since we never call BatchFeature ourselves (we only use
# StreamingDataset, not StreamingDataLoader), the simplest fix is to vendor
# a one-file shim of `transformers.feature_extraction_utils.BatchFeature`
# into the venv's site-packages. If a real transformers is already there
# (e.g. from a previous attempt), remove it first so the shim wins.
echo "==> Removing any existing real transformers install (the shim replaces it)"
uv pip uninstall --python "$VENV_PY" transformers >/dev/null 2>&1 || true

SITE="$("$VENV_PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
TRANSFORMERS_DIR="$SITE/transformers"
mkdir -p "$TRANSFORMERS_DIR"

cat > "$TRANSFORMERS_DIR/__init__.py" <<'PY'
"""Minimal shim of the transformers namespace.

Asparagus + mosaicml-streaming 0.13 only need two symbols from transformers,
both used in an `isinstance(batch, (dict, BatchEncoding, BatchFeature))`
check inside `streaming.base.dataloader.StreamingDataLoader`:

* transformers.feature_extraction_utils.BatchFeature
* transformers.tokenization_utils_base.BatchEncoding

We never use StreamingDataLoader ourselves (we use StreamingDataset), so
these are pure dict subclasses. This shim exists so the eager imports at
the top of streaming.base.dataloader do not fail.
"""

from . import feature_extraction_utils  # noqa: F401
from . import tokenization_utils_base  # noqa: F401

__all__ = ["feature_extraction_utils", "tokenization_utils_base"]
__version__ = "0.0.0+asparagus-shim"
PY

cat > "$TRANSFORMERS_DIR/feature_extraction_utils.py" <<'PY'
"""Stub for `transformers.feature_extraction_utils.BatchFeature`."""

from typing import Any, Optional


class BatchFeature(dict):
    def __init__(self, data: Optional[dict] = None, tensor_type: Any = None):
        super().__init__(data or {})
        self.tensor_type = tensor_type
PY

cat > "$TRANSFORMERS_DIR/tokenization_utils_base.py" <<'PY'
"""Stub for `transformers.tokenization_utils_base.BatchEncoding`."""

from typing import Any, Optional


class BatchEncoding(dict):
    def __init__(
        self,
        data: Optional[dict] = None,
        encoding: Any = None,
        tensor_type: Any = None,
        prepend_batch_axis: bool = False,
        n_sequences: Optional[int] = None,
    ):
        super().__init__(data or {})
        self.encodings = encoding
        self.tensor_type = tensor_type
        self.prepend_batch_axis = prepend_batch_axis
        self.n_sequences = n_sequences
PY

echo "    Wrote shim to $TRANSFORMERS_DIR (BatchFeature + BatchEncoding)"

echo "==> Smoke test"
"$VENV_PY" - <<'PY'
from streaming import StreamingDataset
import streaming
print("OK: streaming.StreamingDataset is importable")
print("streaming.__file__:", streaming.__file__)
PY
