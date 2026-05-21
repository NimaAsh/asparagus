#!/usr/bin/env bash
#
# Install the small subset of mosaicml-streaming needed by the MDS pretraining
# reader into the existing asparagus uv environment.
#
# Run this after:
#   uv sync --group dcai --extra extras
#
# Then launch jobs with ASPARAGUS_SKIP_INSTALL=1 so `uv run --no-sync` keeps
# these manually layered packages in place.
set -euo pipefail

if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: uv is not on PATH." >&2
  exit 127
fi

VENV="${UV_PROJECT_ENVIRONMENT:-/data/$USER/asparagus/venvs/asparagus}"
export UV_PROJECT_ENVIRONMENT="$VENV"
VENV_PY="$VENV/bin/python"

if [[ ! -x "$VENV_PY" ]]; then
  echo "ERROR: $VENV_PY does not exist. Run uv sync first." >&2
  exit 1
fi

echo "==> Target python: $VENV_PY"
"$VENV_PY" -c "import sys, sysconfig; print('exe:', sys.executable); print('site:', sysconfig.get_paths()['purelib'])"

# Pin the version used while validating the FOMO300 MDS reader. Newer releases
# may change eager imports or dependency pins.
uv pip install --python "$VENV_PY" --no-deps mosaicml-streaming==0.13.0
uv pip install --python "$VENV_PY" --no-deps \
  xxhash zstandard zstd Brotli python-snappy cramjam catalogue filelock

SITE="$("$VENV_PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"

# Earlier experiments wrote stub compression modules. Remove those stubs so
# fsspec's import-time compression probes hit the real wheels installed above.
for stub in "$SITE/brotli.py" "$SITE/snappy.py" "$SITE/zstd.py"; do
  if [[ -f "$stub" ]] && grep -q "Stub for" "$stub" 2>/dev/null; then
    rm -f "$stub"
  fi
done
rm -rf "$SITE/__pycache__/brotli.cpython-"*".pyc" \
  "$SITE/__pycache__/snappy.cpython-"*".pyc" \
  "$SITE/__pycache__/zstd.cpython-"*".pyc" 2>/dev/null || true

# streaming.base.dataloader imports two transformers classes even when only
# StreamingDataset is used. A tiny shim avoids pulling in conflicting
# transformers/huggingface-hub/numpy pins.
uv pip uninstall --python "$VENV_PY" transformers >/dev/null 2>&1 || true
TRANSFORMERS_DIR="$SITE/transformers"
mkdir -p "$TRANSFORMERS_DIR"

cat > "$TRANSFORMERS_DIR/__init__.py" <<'PY'
from . import feature_extraction_utils  # noqa: F401
from . import tokenization_utils_base  # noqa: F401

__all__ = ["feature_extraction_utils", "tokenization_utils_base"]
__version__ = "0.0.0+asparagus-shim"
PY

cat > "$TRANSFORMERS_DIR/feature_extraction_utils.py" <<'PY'
class BatchFeature(dict):
    def __init__(self, data=None, tensor_type=None):
        super().__init__(data or {})
        self.tensor_type = tensor_type
PY

cat > "$TRANSFORMERS_DIR/tokenization_utils_base.py" <<'PY'
class BatchEncoding(dict):
    def __init__(self, data=None, encoding=None, tensor_type=None, prepend_batch_axis=False, n_sequences=None):
        super().__init__(data or {})
        self.encodings = encoding
        self.tensor_type = tensor_type
        self.prepend_batch_axis = prepend_batch_axis
        self.n_sequences = n_sequences
PY

STREAMING_INIT="$SITE/streaming/__init__.py"
if [[ -f "$STREAMING_INIT" ]]; then
  STREAMING_INIT_BAK="${STREAMING_INIT}.asparagus-bak"
  if [[ ! -f "$STREAMING_INIT_BAK" ]]; then
    cp "$STREAMING_INIT" "$STREAMING_INIT_BAK"
  fi
  cat > "$STREAMING_INIT" <<'PY'
from streaming.base.dataset import StreamingDataset
from streaming.base.dataloader import StreamingDataLoader

try:
    from streaming.base.stream import Stream
except Exception:
    Stream = None

try:
    from streaming._version import __version__
except Exception:
    __version__ = "0.0.0+asparagus-minimal"

__all__ = ["StreamingDataset", "StreamingDataLoader", "Stream"]
PY
fi

"$VENV_PY" - <<'PY'
import brotli
import lightning  # noqa: F401
import snappy
import zstd
from streaming import StreamingDataset

payload = b"asparagus streaming smoke test"
assert snappy.decompress(snappy.compress(payload)) == payload
assert brotli.decompress(brotli.compress(payload)) == payload
assert zstd.decompress(zstd.compress(payload)) == payload
print("OK: StreamingDataset import works:", StreamingDataset)
PY
