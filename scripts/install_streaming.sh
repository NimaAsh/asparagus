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
# system-library requirement, including bundled native libs for the C
# extensions). We install REAL packages here rather than stubbing because
# fsspec (a lightning transitive dep) probes the compression modules at
# import time with `snappy.compress(b"")` etc., and only catches
# (ImportError, NameError, AttributeError). A stub that raises anything else
# (e.g. NotImplementedError) crashes the whole lightning import chain.
#
#   xxhash         - shard hash validation
#   zstandard      - newer zstd binding (we keep both because some tools
#                    pin to one or the other; both ship pure manylinux wheels).
#   zstd           - legacy zstd binding eagerly imported by
#                    streaming.base.compression.
#   Brotli         - eagerly imported by streaming.base.compression and
#                    probed by fsspec.compression.
#   python-snappy  - eagerly imported by streaming.base.compression and
#                    probed by fsspec.compression. 0.7.x switched its
#                    backend from libsnappy to cramjam, so we install
#                    cramjam too (pure-Rust manylinux wheels, no deps).
#   cramjam        - Rust-based compression backend for python-snappy >=0.7.
#   catalogue      - dataset registry used by streaming.base.registry_utils.
#   filelock       - shard caching lock (usually already installed by
#                    huggingface-hub but explicit-is-better).
uv pip install --python "$VENV_PY" --no-deps \
    xxhash zstandard zstd Brotli python-snappy cramjam catalogue filelock

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

# Earlier versions of this script wrote stub `brotli.py`, `snappy.py`, and
# `zstd.py` files into site-packages. fsspec's compression module probes
# these at import time with `snappy.compress(b"")` and only catches
# (ImportError, NameError, AttributeError), so a stub that raised
# NotImplementedError crashed the whole lightning import chain. We now
# install the real packages above; clean up any leftover stubs so the real
# package's module wins resolution.
echo "==> Cleaning up any pre-existing brotli/snappy/zstd stubs"
for stub in "$SITE/brotli.py" "$SITE/snappy.py" "$SITE/zstd.py"; do
    if [[ -f "$stub" ]]; then
        # Detect our stub by its docstring. Don't delete a legitimate
        # single-file module that pip put there.
        if grep -q "Stub for the" "$stub" 2>/dev/null; then
            rm -f "$stub"
            echo "    Removed stub: $stub"
        fi
    fi
done
# Also clear cached bytecode for the stubs so the next import resolves the
# real packages cleanly.
rm -rf \
    "$SITE/__pycache__/brotli.cpython-"*".pyc" \
    "$SITE/__pycache__/snappy.cpython-"*".pyc" \
    "$SITE/__pycache__/zstd.cpython-"*".pyc" 2>/dev/null || true

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
# Order matters: lightning -> fsspec probes the compression modules
# (snappy.compress(b""), zstd.compress(b"")). If our compression installs
# regressed (e.g. left a stub behind), the lightning import below will fail
# the same way the SLURM job did. We exercise that path here so the script
# fails loudly on a login node instead of silently in a job.
import lightning  # noqa: F401
import fsspec  # noqa: F401
import snappy
import brotli
import zstd
assert snappy.compress(b"") == b"", "snappy.compress(b'') must roundtrip"
assert brotli.compress(b"x") and len(brotli.compress(b"x")) > 0, "brotli broken"
assert zstd.compress(b"x"), "zstd broken"

from streaming import StreamingDataset
import streaming
print("OK: lightning + fsspec import cleanly")
print("OK: snappy/brotli/zstd compress() are real")
print("OK: streaming.StreamingDataset is importable")
print("streaming.__file__:", streaming.__file__)
PY
