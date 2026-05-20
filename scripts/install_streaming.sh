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
# Real runtime deps of `streaming.base.*` that have no version conflicts with
# asparagus' pins (these are pure-python or have manylinux wheels with no
# system-library requirement):
#   xxhash     - shard hash validation
#   zstandard  - kept for parity (not the same module as `zstd` below; we stub
#                that one because `zstd` PyPI package is the legacy binding
#                streaming.base.compression eagerly imports).
#   catalogue  - dataset registry used by streaming.base.registry_utils
#   filelock   - shard caching lock (almost always already installed by huggingface-hub)
uv pip install --python "$VENV_PY" --no-deps xxhash zstandard catalogue filelock

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

# `streaming/base/compression.py` does `import brotli; import snappy; import zstd`
# at module load time. We never write or decompress shards with these
# algorithms (our FOMO300 shards are uncompressed), so installing real
# packages just to satisfy the imports is wasteful and `python-snappy`
# additionally needs a system libsnappy-dev that isn't on every node.
# Stub them out as no-op modules with compress/decompress functions that
# raise if anyone actually calls them.
echo "==> Writing compression shims (brotli, snappy, zstd)"
for mod in brotli snappy zstd; do
    cat > "$SITE/${mod}.py" <<PY
"""Stub for the '$mod' module so streaming.base.compression can import.

We only read uncompressed MDS shards; this module's compress()/decompress()
are never called by our code path. They raise to make accidental use loud.
"""


def compress(data, *args, **kwargs):
    raise NotImplementedError(
        "$mod is a stubbed module in this env (no real compression library). "
        "Install the real package if you need shard (de)compression."
    )


def decompress(data, *args, **kwargs):
    raise NotImplementedError(
        "$mod is a stubbed module in this env (no real compression library). "
        "Install the real package if you need shard (de)compression."
    )
PY
done
echo "    Wrote $SITE/{brotli,snappy,zstd}.py"

# `streaming/__init__.py` eagerly imports streaming.multimodal, streaming.text,
# streaming.vision. text.c4 imports `transformers.models.auto.tokenization_auto`
# (not in our shim), vision imports PIL/torchvision-internal stuff, etc. None
# of those subpackages are needed for plain `StreamingDataset` over local MDS
# shards. Replace the top-level __init__.py with a minimal version that just
# re-exports the symbols we (or anything that imports our module) use.
STREAMING_INIT="$SITE/streaming/__init__.py"
if [[ -f "$STREAMING_INIT" ]]; then
    STREAMING_INIT_BAK="${STREAMING_INIT}.asparagus-bak"
    if [[ ! -f "$STREAMING_INIT_BAK" ]]; then
        cp "$STREAMING_INIT" "$STREAMING_INIT_BAK"
        echo "==> Backed up original streaming/__init__.py to $STREAMING_INIT_BAK"
    fi
    cat > "$STREAMING_INIT" <<'PY'
"""Minimal streaming/__init__.py for asparagus pretraining.

The upstream version eagerly imports streaming.multimodal, streaming.text,
and streaming.vision, which transitively require AutoTokenizer, PIL, and
several other packages we do not install. We only need StreamingDataset
(and StreamingDataLoader for API surface), so re-export just those.

Original file is preserved at __init__.py.asparagus-bak.
"""

from streaming.base.dataset import StreamingDataset
from streaming.base.dataloader import StreamingDataLoader

try:
    from streaming._version import __version__
except Exception:
    __version__ = "0.0.0+asparagus-minimal"

__all__ = ["StreamingDataset", "StreamingDataLoader"]
PY
    echo "==> Rewrote $STREAMING_INIT (multimodal/text/vision skipped)"
fi

echo "==> Smoke test"
"$VENV_PY" - <<'PY'
from streaming import StreamingDataset
import streaming
print("OK: streaming.StreamingDataset is importable")
print("streaming.__file__:", streaming.__file__)
PY
