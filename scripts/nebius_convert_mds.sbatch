#!/usr/bin/env bash
#SBATCH --job-name=asp-convert-mds
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --partition=main
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

mkdir -p "$OUTPUT_ROOT"

echo "Job id: ${SLURM_JOB_ID:-local}"
echo "Node list: ${SLURM_JOB_NODELIST:-local}"
echo "Repo: $REPO_DIR"
echo "MDS root: $MDS_ROOT"
echo "Output root: $OUTPUT_ROOT"
echo "Task name: $TASK_NAME"
echo "Split name: $SPLIT_NAME"

cd "$REPO_DIR"

if command -v uv >/dev/null 2>&1; then
  export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$PROJECT_DIR/venvs/asparagus}"
  uv run --with mosaicml-streaming python asparagus/scripts/mds_to_asparagus_pretrain.py \
    --mds-root "$MDS_ROOT" \
    --output-root "$OUTPUT_ROOT" \
    --task-name "$TASK_NAME" \
    --split-name "$SPLIT_NAME" \
    "$@"
else
  python -m pip install mosaicml-streaming
  python asparagus/scripts/mds_to_asparagus_pretrain.py \
    --mds-root "$MDS_ROOT" \
    --output-root "$OUTPUT_ROOT" \
    --task-name "$TASK_NAME" \
    --split-name "$SPLIT_NAME" \
    "$@"
fi
