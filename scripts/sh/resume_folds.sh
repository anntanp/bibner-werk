#!/usr/bin/env bash
# Purpose:      Wait for any running train_gliner_silver.py fold jobs (owned by
#               the current user) to finish, then relaunch every fold that
#               doesn't have a clean checkpoint — missing "Saved checkpoint" in
#               its log, or any "Skipping batch due to CUDA OOM" warning
#               (silently dropped training batches; treated as compromised,
#               not just slow, since a k-fold variance estimate needs
#               comparable training conditions across replicates) — capped at
#               a configurable concurrency limit. Written after a real
#               incident (2026-07-10): on a shared A100 with another user's
#               jobs holding a fixed ~19 GiB, launching all 10 k-fold
#               replicates at once caused cascading OOM kills across 9 of 10
#               folds. See notes/fiz/bibner-werk/A5-finetuning-gliner-actual.md
#               #16 and notes/fiz/bibner-werk/A5-finetuning-workflow.md §7.
# Usage:        bash scripts/sh/resume_folds.sh [-k FOLD_COUNT] [-c CONCURRENCY]
#                 -k: number of fold replicates, 0..k-1 (default: 10)
#                 -c: max concurrent training processes (default: 3 — sized
#                     for an 80GB A100 with ~19GB already claimed by another
#                     tenant's job; recompute for a different GPU/tenancy
#                     situation, see the incident writeup above)
# Inputs:       fold{i}_full.log for i in 0..k-1, in the repo root (produced
#                 by a prior `train_gliner_silver.py --fold-index i | tee
#                 fold{i}_full.log` run)
# Outputs:      fold{i}_full.log for every relaunched fold (overwritten);
#                 models/gliner_fold{i}/ checkpoints (via train_gliner_silver.py)
# Dependencies: pgrep, xargs, python3, scripts/py/train_gliner_silver.py
# Assumptions:  Run from the repo root (paths below are relative to it). Only
#               waits on and checks for the current user's own
#               train_gliner_silver.py processes — never touches or waits on
#               another tenant's GPU jobs.
set -euo pipefail

FOLD_COUNT=10
CONCURRENCY=3

while getopts "k:c:" opt; do
  case "$opt" in
    k) FOLD_COUNT="$OPTARG" ;;
    c) CONCURRENCY="$OPTARG" ;;
    *) echo "Usage: $0 [-k FOLD_COUNT] [-c CONCURRENCY]" >&2; exit 1 ;;
  esac
done

echo "$(date): waiting for existing train_gliner_silver.py runs (user $USER) to finish..."
while pgrep -u "$USER" -f train_gliner_silver.py > /dev/null; do
  sleep 60
done
echo "$(date): GPU clear of your training jobs."

RERUN=()
for i in $(seq 0 $((FOLD_COUNT - 1))); do
  log="fold${i}_full.log"
  if ! grep -q "Saved checkpoint" "$log" 2>/dev/null \
     || [ "$(grep -c "Skipping batch due to CUDA OOM" "$log" 2>/dev/null || true)" -gt 0 ]; then
    RERUN+=("$i")
  fi
done
echo "$(date): folds needing (re)run: ${RERUN[*]:-none}"

if [ "${#RERUN[@]}" -eq 0 ]; then
  echo "$(date): all $FOLD_COUNT folds already have clean checkpoints. Nothing to do."
  exit 0
fi

# Any fold in RERUN has no trustworthy checkpoint by definition (that's the criterion above), so its
# output dir may still hold stale checkpoint-N/ subdirs from save_strategy="epoch" on the killed/crashed
# attempt — clear it before relaunching rather than risk a fresh run's own checkpoint-N colliding with
# (or being mistaken for) the old one. Never touches folds NOT in RERUN.
for i in "${RERUN[@]}"; do
  rm -rf "models/gliner_fold${i}"
done
echo "$(date): cleared stale output dirs for: ${RERUN[*]}"

printf '%s\n' "${RERUN[@]}" | xargs -P "$CONCURRENCY" -I{} bash -c \
  'time python3 scripts/py/train_gliner_silver.py --fold-index {} 2>&1 | tee "fold{}_full.log"'

echo "$(date): resume run complete."
