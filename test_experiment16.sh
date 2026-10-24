#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
    echo 'usage: bash test_experiment16.sh checkpoint output_dir [maps|fuse]' >&2
    exit 2
fi
CHECKPOINT="$1"
OUT_DIR="$2"
RUN_MODE="${3:-maps}"
case "$RUN_MODE" in maps|fuse) ;; *) exit 2 ;; esac
TEST_SCANS="${SGER_TEST_LIST:-lists/dtu/test.txt}"
COMMAND=(python -u test.py --dataset general_eval
    --testpath "${TESTPATH:-data/DTU/dtu_test_all}" --testlist "$TEST_SCANS"
    --loadckpt "$CHECKPOINT" --outdir "$OUT_DIR/paired"
    --interval_scale 1.06 --ndepths 48,32,8 --depth_inter_r 4,2,1
    --use_sger_lite --sger_proposal_gate --sger_depth_normalization batch
    --sger_max_residual_ratio 0.25 --batch_size 1 --num_view 5
    --max_h 864 --max_w 1152 --save_only --save_freq 1000)
FUSION=(python -u experiment15_e_amplitude.py
    --source_dir "$OUT_DIR/paired" --output_dir "$OUT_DIR/fusion"
    --scan_list "$TEST_SCANS" --alphas 0,1 --fuse
    --fusibile_exe_path "${FUSIBILE_EXE:-../../fusibile-master/build/fusibile}"
    --prob_threshold 0.9 --disp_threshold 0.25 --num_consistent 4)
if [ "${DRY_RUN:-0}" = 1 ]; then
    printf '%q ' "${COMMAND[@]}"
    printf '\n'
    if [ "$RUN_MODE" = fuse ]; then printf '%q ' "${FUSION[@]}"; printf '\n'; fi
else
    export OMP_NUM_THREADS=1
    export PYTHONPATH=.
    "${COMMAND[@]}"
    if [ "$RUN_MODE" = fuse ]; then
        "${FUSION[@]}"
    fi
fi
