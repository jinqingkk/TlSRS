#!/usr/bin/env bash
set -euo pipefail

# Formal end-to-end training for Experiment 15-E.  Run through bash because
# repository shell files are not guaranteed to have executable permission.
LOG_DIR="${1:-./checkpoints/experiment15_e_formal_stable_bce}"
RUN_MODE="${2:-start}"
MVS_TRAINING_ROOT="${MVS_TRAINING:-./data/DTU/mvs_training/dtu/}"

case "$RUN_MODE" in
    start|resume) ;;
    *)
        echo "usage: bash train_experiment15_e.sh [log_dir] [start|resume]" >&2
        exit 2
        ;;
esac

if [ ! -d "$MVS_TRAINING_ROOT" ]; then
    echo "DTU training root does not exist: $MVS_TRAINING_ROOT" >&2
    echo "Set MVS_TRAINING to the DTU training directory." >&2
    exit 2
fi

shopt -s nullglob
EXISTING_CHECKPOINTS=("$LOG_DIR"/model_*.ckpt)
RESUME_ARGS=()
if [ "$RUN_MODE" = "start" ]; then
    if [ "${#EXISTING_CHECKPOINTS[@]}" -ne 0 ]; then
        echo "Refusing to start over an existing checkpoint directory: $LOG_DIR" >&2
        exit 2
    fi
else
    if [ "${#EXISTING_CHECKPOINTS[@]}" -eq 0 ]; then
        echo "Cannot resume: no model_*.ckpt exists in $LOG_DIR" >&2
        exit 2
    fi
    RESUME_ARGS=(--resume)
fi

COMMAND=(
    python -m torch.distributed.launch --nproc_per_node=1
    train.py
    --logdir "$LOG_DIR"
    --dataset dtu_yao
    --trainpath "$MVS_TRAINING_ROOT"
    --trainlist lists/dtu/train.txt
    --testlist lists/dtu/test.txt
    --numdepth 192
    --interval_scale 1.06
    --ndepths 48,32,8
    --depth_inter_r 4,2,1
    --dlossw 0.5,1.0,2.0
    --cr_base_chs 8,8,8
    --batch_size 2
    --epochs 16
    --lr 0.001
    --lrepochs 10,12,14:2
    --wd 0.0
    --eval_freq 3
    --save_freq 1
    --summary_freq 50
    --seed 1
    --use_sger_lite
    --sger_max_residual_ratio 0.25
    --sger_residual_final_activation relu
    --sger_residual_negative_slope 0.01
    --raw_depth_loss_weight 1.0
    --refined_depth_loss_weight 1.0
    --residual_loss_weight 0.01
    --gate_loss_weight 0.0
    --safe_refine_loss_weight 0.1
    --safe_refine_margin 0.0
    --residual_target_loss_weight 0.05
    --gate_benefit_loss_weight 0.05
    --gate_bce_mode stable_logprob
    --residual_target_ratio 0.25
    --benefit_margin_ratio 0.05
    --freeze_backbone_epochs 0
    --backbone_lr_scale 0.1
    --sger_warmup_start_epoch 7
    --sger_warmup_end_epoch 10
    --normal_smooth_loss_weight 0.02
    --curv_loss_weight 0.005
    --edge_smooth_loss_weight 0.005
    --depth_normal_loss_weight 0.03
    --depth_normal_conf_threshold 0.8
    --edge_grad_threshold 0.05
    --geometry_conf_mid 0.65
    --geometry_k_conf 10.0
    --geometry_edge_mid 0.25
    --geometry_k_edge 10.0
    --geometry_w_min 0.05
    --region_lambda_a 1.5
    --region_lambda_b 1.0
    --region_edge_threshold 0.25
    --region_depth_threshold 0.02
    --region_curv_threshold 0.02
    --region_smooth_k 2.0
    "${RESUME_ARGS[@]}"
)

if [ "${DRY_RUN:-0}" = "1" ]; then
    printf '%q ' "${COMMAND[@]}"
    printf '\n'
    exit 0
fi

mkdir -p "$LOG_DIR"
{
    printf 'experiment=15-E formal stable-BCE SGER-lite training\n'
    printf 'run_mode=%s\n' "$RUN_MODE"
    printf 'started_at=%s\n' "$(date --iso-8601=seconds)"
    printf 'working_directory=%s\n' "$PWD"
    printf 'command='
    printf '%q ' "${COMMAND[@]}"
    printf '\n'
    sha256sum train.py models/cas_mvsnet.py models/module.py \
        models/sger_refinement.py train_experiment15_e.sh
} | tee -a "$LOG_DIR/formal_training_manifest.txt"

CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 PYTHONPATH=. \
    "${COMMAND[@]}" 2>&1 | tee -a "$LOG_DIR/log.txt"

