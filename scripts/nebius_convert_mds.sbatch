#!/usr/bin/env bash
#SBATCH --job-name=asp-convert-mds
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --partition=n
#SBATCH --account=training
#SBATCH --qos=low
#SBATCH --output=slurms/%j.out
#SBATCH --error=slurms/%j.err

set -euo pipefail

mkdir -p slurms

REPO_DIR="${REPO_DIR:-$PWD}"
PROJECT_DIR="${PROJECT_DIR:-/data/$USER/asparagus}"
MDS_ROOT="${MDS_ROOT:-/data/mihir-stuff/FOMO300/mds}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_DIR/data}"
TASK_NAME="${TASK_NAME:-PT901_FOMO300_MDS}"
SPLIT_NAME="${SPLIT_NAME:-split_mihir_90_10}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$PROJECT_DIR/uv-cache}"

mkdir -p "$OUTPUT_ROOT" "$UV_CACHE_DIR"

echo "Job id: ${SLURM_JOB_ID:-local}"
echo "Node list: ${SLURM_JOB_NODELIST:-local}"
echo "Repo: $REPO_DIR"
echo "MDS root: $MDS_ROOT"
echo "Output root: $OUTPUT_ROOT"
echo "Task name: $TASK_NAME"
echo "Split name: $SPLIT_NAME"

cd "$REPO_DIR"

if command -v uv >/dev/null 2>&1; then
  export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$PROJECT_DIR/venvs/mds-convert}"
  uv run --no-project \
    --with torch==2.6.0 \
    --with mosaicml-streaming \
    python asparagus/scripts/mds_to_asparagus_pretrain.py \
    --mds-root "$MDS_ROOT" \
    --output-root "$OUTPUT_ROOT" \
    --task-name "$TASK_NAME" \
    --split-name "$SPLIT_NAME" \
    "$@"
else
  PYTHON_BIN="${PYTHON_BIN:-}"
  if [[ -z "$PYTHON_BIN" ]]; then
    PYTHON_BIN="$(command -v python3 || command -v python || true)"
  fi
  if [[ -z "$PYTHON_BIN" ]]; then
    echo "ERROR: neither uv nor python/python3 is available on PATH." >&2
    echo "Load a Python module, install uv, or run this job in a Python container." >&2
    exit 127
  fi

  "$PYTHON_BIN" -m pip install mosaicml-streaming
  "$PYTHON_BIN" asparagus/scripts/mds_to_asparagus_pretrain.py \
    --mds-root "$MDS_ROOT" \
    --output-root "$OUTPUT_ROOT" \
    --task-name "$TASK_NAME" \
    --split-name "$SPLIT_NAME" \
    "$@"
fi
