#!/usr/bin/env bash
#SBATCH --job-name=asp-pretrain
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=16
#SBATCH --gpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --partition=n
#SBATCH --account=training
#SBATCH --qos=low
#SBATCH --requeue
#SBATCH --output=slurms/%j.out
#SBATCH --error=slurms/%j.err

set -euo pipefail

mkdir -p slurms

CONFIG_NAME="${1:-projects/fomo26/pretrain/resenc_fomo300_mds_short}"
if [[ $# -gt 0 ]]; then
  shift
fi

REPO_DIR="${REPO_DIR:-$PWD}"
PROJECT_DIR="${PROJECT_DIR:-/data/$USER/asparagus}"

export ASPARAGUS_CONFIGS="${ASPARAGUS_CONFIGS:-$REPO_DIR/configs}"
export ASPARAGUS_DATA="${ASPARAGUS_DATA:-$PROJECT_DIR/data}"
export ASPARAGUS_MODELS="${ASPARAGUS_MODELS:-$PROJECT_DIR/models}"
export ASPARAGUS_RESULTS="${ASPARAGUS_RESULTS:-$PROJECT_DIR/results}"
export ASPARAGUS_RAW_LABELS="${ASPARAGUS_RAW_LABELS:-$PROJECT_DIR/raw_labels}"
export WANDB_DIR="${WANDB_DIR:-$PROJECT_DIR/wandb}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$PROJECT_DIR/uv-cache}"

mkdir -p \
  "$ASPARAGUS_DATA" \
  "$ASPARAGUS_MODELS" \
  "$ASPARAGUS_RESULTS" \
  "$ASPARAGUS_RAW_LABELS" \
  "$WANDB_DIR" \
  "$UV_CACHE_DIR"

echo "Job id: ${SLURM_JOB_ID:-local}"
echo "Node list: ${SLURM_JOB_NODELIST:-local}"
echo "Config: $CONFIG_NAME"
echo "Repo: $REPO_DIR"
echo "Project dir: $PROJECT_DIR"
echo "ASPARAGUS_DATA: $ASPARAGUS_DATA"
echo "ASPARAGUS_MODELS: $ASPARAGUS_MODELS"

run_asparagus() {
  cd "$REPO_DIR"

  if command -v uv >/dev/null 2>&1; then
    export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$PROJECT_DIR/venvs/asparagus}"
    if [[ "${ASPARAGUS_SKIP_INSTALL:-0}" != "1" ]]; then
      uv sync --group dcai --extra extras
    fi
    uv run asp_pretrain --config-name "$CONFIG_NAME" "$@"
  else
    if [[ "${ASPARAGUS_SKIP_INSTALL:-0}" != "1" ]]; then
      python -m pip install -e ".[extras]"
    fi
    asp_pretrain --config-name "$CONFIG_NAME" "$@"
  fi
}

export -f run_asparagus

if [[ -n "${ASPARAGUS_CONTAINER_IMAGE:-}" && -z "${ASPARAGUS_INSIDE_CONTAINER:-}" ]]; then
  export ASPARAGUS_INSIDE_CONTAINER=1
  srun \
    --container-image="$ASPARAGUS_CONTAINER_IMAGE" \
    --container-mounts="$REPO_DIR:$REPO_DIR,/data:/data,/block:/block" \
    --container-workdir="$REPO_DIR" \
    bash -lc 'run_asparagus "$@"' bash "$@"
else
  run_asparagus "$@"
fi
