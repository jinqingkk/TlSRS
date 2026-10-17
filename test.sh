#!/usr/bin/env bash
TESTPATH="data/DTU/dtu_test_all"
TESTLIST="lists/dtu/test.txt"
CKPT_FILE=$1
shift

python test.py \
  --dataset=general_eval \
  --batch_size=1 \
  --testpath="$TESTPATH" \
  --testlist="$TESTLIST" \
  --loadckpt "$CKPT_FILE" \
  "$@"
