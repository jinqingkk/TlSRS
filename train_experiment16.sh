#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# A new gate adaptation experiment from the trained 15-E model, not from zero.
OUT_DIR="${1:-checkpoints/experiment16_proposal_gate}"
if [ "$#" -gt 0 ]; then shift; fi
COMMAND=(python -u train_sger_gate.py
    --checkpoint "${SGER_INIT_CHECKPOINT:-checkpoints/experiment15_e_formal_stable_bce/model_000012.ckpt}"
    --trainpath "${MVS_TRAINING:-data/DTU/mvs_training/dtu}"
    --output_dir "$OUT_DIR" --variants proposal
    --epochs 1 --validation all --batch_size 2 --lr 0.0001 --seed 1 "$@")
if [ "${DRY_RUN:-0}" = 1 ]; then
    printf '%q ' "${COMMAND[@]}"
    printf '\n'
else
    export OMP_NUM_THREADS=1
    export PYTHONPATH=.
    "${COMMAND[@]}"
fi
