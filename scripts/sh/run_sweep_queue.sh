#!/usr/bin/env bash
# Purpose:      Sequentially fire this GPU's assigned slice of the data-scaling sweep
#               (A5-finetuning-scaling-sweep-plan.md checklist §4: 1k/10k/20k, both
#               models, k=10). Simplified single-GPU sequential queue, not the full
#               concurrency-cap/timeout job-queue runner originally scoped in §3 —
#               proportionate for this 60-job scope given deadline pressure
#               (jcdl-plan-202606.md §1). Relies on each training script's own
#               preflight checks + resume-from-checkpoint (already built) for safety,
#               not on anything in this script.
# Usage:        CUDA_VISIBLE_DEVICES=<n> ./scripts/sh/run_sweep_queue.sh <precision> <queue_file> <log_file>
#               e.g. CUDA_VISIBLE_DEVICES=0 ./scripts/sh/run_sweep_queue.sh bf16 \
#                    data/eval/sweep_queue_gpu0.csv data/eval/sweep_status_gpu0.log
# Inputs:       queue_file: CSV, one job per line, "model,train_size,fold_index"
#                   (model is "xlmr" or "gliner")
# Outputs:      log_file: one line per job, "TIMESTAMP MODEL SIZE FOLD STATUS"
#                   (STARTED/COMPLETED/FAILED); models/{xlmr,gliner}_n{size}_fold{i}/
#                   checkpoints via the training scripts themselves
# Dependencies: train_xlmr_silver.py, train_gliner_silver.py (must be run from repo root)
# Assumptions:  CUDA_VISIBLE_DEVICES must be set by the caller before invoking this
#               script (one GPU per invocation — never leave it unset, see the
#               documented DataParallel-hang incident, A5-finetuning-xlmr-workflow.md
#               §5.2 item 0). No interactive prompts anywhere in this script or the
#               training scripts' cold path, safe for unattended multi-hour operation.
#               A job that fails (non-zero exit) is logged and the queue continues to
#               the next job, not aborted — a single bad job shouldn't block the rest
#               of this GPU's assigned work overnight.
set -uo pipefail

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  echo "ERROR: CUDA_VISIBLE_DEVICES must be set explicitly before running this script." >&2
  exit 1
fi

precision="${1:?Usage: run_sweep_queue.sh <precision> <queue_file> <log_file>}"
queue_file="${2:?Usage: run_sweep_queue.sh <precision> <queue_file> <log_file>}"
log_file="${3:?Usage: run_sweep_queue.sh <precision> <queue_file> <log_file>}"

mkdir -p "$(dirname "$log_file")"

while IFS=',' read -r model size fold; do
  [ -z "$model" ] && continue
  ts=$(date '+%Y-%m-%d %H:%M:%S')
  echo "$ts $model $size $fold STARTED" | tee -a "$log_file"

  if [ "$model" = "xlmr" ]; then
    script="scripts/py/train_xlmr_silver.py"
  elif [ "$model" = "gliner" ]; then
    script="scripts/py/train_gliner_silver.py"
  else
    echo "$ts $model $size $fold FAILED (unknown model)" | tee -a "$log_file"
    continue
  fi

  if python3 "$script" --fold-index "$fold" --train-size "$size" --precision "$precision"; then
    ts=$(date '+%Y-%m-%d %H:%M:%S')
    echo "$ts $model $size $fold COMPLETED" | tee -a "$log_file"
  else
    ts=$(date '+%Y-%m-%d %H:%M:%S')
    echo "$ts $model $size $fold FAILED (exit $?)" | tee -a "$log_file"
  fi
done < "$queue_file"

echo "$(date '+%Y-%m-%d %H:%M:%S') QUEUE DONE ($queue_file)" | tee -a "$log_file"
