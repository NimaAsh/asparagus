#!/usr/bin/env bash
# DDP-aware sbatch wrapper for asp_pretrain on Nebius. Defaults to one full
# H100 node (8 GPUs, 8 ranks, srun-launched), but the SBATCH directives can
# be overridden at submit time:
#
#   sbatch --gres=gpu:4 --ntasks-per-node=4 scripts/nebius_pretrain_ddp.sbatch ...
#
# The key differences vs nebius_pretrain.sbatch (the 1-GPU launcher):
#   1. --ntasks-per-node matches --gres=gpu:N. SLURM spawns one task per GPU
#      and Lightning's SLURMEnvironment auto-detects rank/world-size from
#      SLURM_PROCID / SLURM_NTASKS. The single-task launcher uses
#      Lightning's SubprocessScriptLauncher, which is incompatible with
#      DDP under SLURM (validate_settings refuses devices>ntasks).
#   2. We invoke python via srun so SLURM actually launches all N tasks.
#   3. We pin ASPARAGUS_RUN_ID=$SLURM_JOB_ID so every rank's `${version:}`
#      hydra resolver returns the same value (otherwise each rank picks a
#      random unused int and ends up in a different output dir).
#
# NOTE: We intentionally do NOT set --gpus-per-task. With SLURM cgroup GPU
# pinning, --gpus-per-task=1 makes each task see exactly one CUDA device,
# but Lightning's accelerator validates `devices=8` against
# torch.cuda.device_count() *inside* the per-rank process and fails with
# "You requested gpu: [0..7] But your machine only has: [0]". Leaving the
# GPUs at the job level (`--gres=gpu:8`) lets all 8 ranks see all 8 GPUs,
# and Lightning's DDPStrategy picks each rank's device from LOCAL_RANK.
#SBATCH --job-name=asp-pretrain-ddp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:8
# Multi-GPU runs do not always achieve linear scaling on a data-bound
# workload, so leave a generous wall budget by default. AGENTS.md sanctions
# `--time=infinite` for training jobs because the Nebius nodes are reserved.
# Pass `--time=...` on the sbatch CLI when you want a shorter cap.
#SBATCH --time=infinite
#SBATCH --partition=n
#SBATCH --account=training
#SBATCH --qos=normal
#SBATCH --requeue
#SBATCH --output=slurms/%j.out
#SBATCH --error=slurms/%j.err

set -euo pipefail

mkdir -p slurms

CONFIG_NAME="${1:-projects/fomo26/pretrain/unet_m_fomo300_mds_short_fast_8gpu}"
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

# Pin run_id so all ranks resolve `${version:}` identically. See _resolve_version
# in asparagus/pipeline/run/pretrain.py.
export ASPARAGUS_RUN_ID="${SLURM_JOB_ID:-local}"

# Ensure every rank sees every allocated GPU. Lightning's DDPStrategy
# selects the rank's actual device via LOCAL_RANK; if cgroup pinning
# restricts visibility to one GPU per task, the accelerator parse step
# fails before DDP setup. Forcing CUDA_VISIBLE_DEVICES here is a no-op on
# unconstrained clusters and the explicit fix on constrained ones.
GPUS_ON_NODE="${SLURM_GPUS_ON_NODE:-8}"
export CUDA_VISIBLE_DEVICES="$(python3 -c "print(','.join(str(i) for i in range(${GPUS_ON_NODE})))")"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

mkdir -p \
  "$ASPARAGUS_DATA" \
  "$ASPARAGUS_MODELS" \
  "$ASPARAGUS_RESULTS" \
  "$ASPARAGUS_RAW_LABELS" \
  "$WANDB_DIR" \
  "$UV_CACHE_DIR"

echo "Job id: ${SLURM_JOB_ID:-local}"
echo "Node list: ${SLURM_JOB_NODELIST:-local}"
echo "Tasks per node: ${SLURM_NTASKS_PER_NODE:-1}"
echo "CPUs per task: ${SLURM_CPUS_PER_TASK:-?}"
echo "GPUs (job): ${SLURM_GPUS_ON_NODE:-?}"
echo "Config: $CONFIG_NAME"
echo "Repo: $REPO_DIR"
echo "Project dir: $PROJECT_DIR"
echo "ASPARAGUS_DATA: $ASPARAGUS_DATA"
echo "ASPARAGUS_MODELS: $ASPARAGUS_MODELS"
echo "ASPARAGUS_RUN_ID: $ASPARAGUS_RUN_ID"

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

# uv sync runs ONCE in the parent shell (before srun) so the 8 ranks don't
# race to write to the same venv. Skippable when the env is known to be in
# sync via ASPARAGUS_SKIP_INSTALL=1, which we recommend after the first run
# of the job to keep launch latency low.
if command -v uv >/dev/null 2>&1; then
  export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$PROJECT_DIR/venvs/asparagus}"
  cd "$REPO_DIR"
  if [[ "${ASPARAGUS_SKIP_INSTALL:-0}" != "1" ]]; then
    uv sync --group dcai --extra extras
  fi
fi

start_telemetry

cd "$REPO_DIR"

# --kill-on-bad-exit=1: if any rank dies the whole step is torn down quickly
# instead of leaving 7 ranks hanging on collective ops.
# --gpu-bind=none: disables srun-level GPU binding so CUDA_VISIBLE_DEVICES
# (set above to all 8) survives into each task; Lightning then picks the
# right device from LOCAL_RANK.
if command -v uv >/dev/null 2>&1; then
  srun --kill-on-bad-exit=1 --gpu-bind=none \
    uv run asp_pretrain --config-name "$CONFIG_NAME" "$@"
else
  srun --kill-on-bad-exit=1 --gpu-bind=none \
    asp_pretrain --config-name "$CONFIG_NAME" "$@"
fi
