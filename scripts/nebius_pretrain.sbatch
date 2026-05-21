#!/usr/bin/env bash
#SBATCH --job-name=asp-pretrain
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=16
#SBATCH --gpus-per-task=1
#SBATCH --time=00:45:00
#SBATCH --partition=n
#SBATCH --account=training
# `normal` priority by default; pass `--qos=low` on the sbatch CLI to be
# polite when the cluster is busy or when running long throwaway probes.
#SBATCH --qos=normal
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

# Prefer the checkout this job was submitted from over any stale editable
# install that may point at a different worktree.
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"

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

TELEMETRY_PREFIX="slurms/${SLURM_JOB_ID:-local}"
TELEMETRY_INTERVAL="${ASPARAGUS_TELEMETRY_INTERVAL:-5}"
TELEMETRY_PIDS=()

start_telemetry() {
  if [[ "${ASPARAGUS_TELEMETRY:-1}" != "1" ]]; then
    echo "Telemetry disabled"
    return
  fi

  echo "Telemetry prefix: $TELEMETRY_PREFIX"
  echo "Telemetry interval: ${TELEMETRY_INTERVAL}s"

  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,power.limit,clocks.sm,clocks.mem,temperature.gpu --format=csv,nounits -l "$TELEMETRY_INTERVAL" \
      > "${TELEMETRY_PREFIX}_gpu.csv" 2> "${TELEMETRY_PREFIX}_gpu.err" &
    TELEMETRY_PIDS+=("$!")

    nvidia-smi dmon -s pucvmt -d "$TELEMETRY_INTERVAL" -o DT \
      > "${TELEMETRY_PREFIX}_gpu_dmon.log" 2> "${TELEMETRY_PREFIX}_gpu_dmon.err" &
    TELEMETRY_PIDS+=("$!")

    nvidia-smi topo -m > "${TELEMETRY_PREFIX}_gpu_topo.txt" 2> "${TELEMETRY_PREFIX}_gpu_topo.err" || true
  else
    echo "nvidia-smi not found; GPU telemetry disabled"
  fi

  if command -v vmstat >/dev/null 2>&1; then
    vmstat -t "$TELEMETRY_INTERVAL" \
      > "${TELEMETRY_PREFIX}_vmstat.log" 2> "${TELEMETRY_PREFIX}_vmstat.err" &
    TELEMETRY_PIDS+=("$!")
  else
    echo "vmstat not found; CPU/memory telemetry disabled"
  fi

  if command -v iostat >/dev/null 2>&1; then
    iostat -xz -t "$TELEMETRY_INTERVAL" \
      > "${TELEMETRY_PREFIX}_iostat.log" 2> "${TELEMETRY_PREFIX}_iostat.err" &
    TELEMETRY_PIDS+=("$!")
  else
    echo "iostat not found; disk telemetry disabled"
  fi
}

stop_telemetry() {
  if [[ "${#TELEMETRY_PIDS[@]}" -gt 0 ]]; then
    kill "${TELEMETRY_PIDS[@]}" >/dev/null 2>&1 || true
  fi
}

trap stop_telemetry EXIT

run_asparagus() {
  cd "$REPO_DIR"

  if command -v uv >/dev/null 2>&1; then
    export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$PROJECT_DIR/venvs/asparagus}"
    UV_RUN_ARGS=()
    if [[ "${ASPARAGUS_SKIP_INSTALL:-0}" != "1" ]]; then
      uv sync --group dcai --extra extras
    else
      # Keep manually layered packages such as mosaicml-streaming.
      UV_RUN_ARGS+=("--no-sync")
    fi
    uv run "${UV_RUN_ARGS[@]}" asp_pretrain --config-name "$CONFIG_NAME" "$@"
  else
    if [[ "${ASPARAGUS_SKIP_INSTALL:-0}" != "1" ]]; then
      python -m pip install -e ".[extras]"
    fi
    asp_pretrain --config-name "$CONFIG_NAME" "$@"
  fi
}

export -f run_asparagus

start_telemetry

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
