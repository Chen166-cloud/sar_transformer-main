#!/usr/bin/env bash
# Target-L eval sweep on main ckpt: stop reverse at L(t*)~L_out and record ratio stats.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

CKPT="weights/combo_v1_L1_ckpt_0015000.pt"
for Lout in 2 4 16 66 266 10000; do
    target="eval/results/target_L_${Lout}.csv"
    if [ -f "$target" ]; then echo "SKIP $target"; continue; fi
    CUDA_VISIBLE_DEVICES=0 python3 eval/run_eval.py \
        --ckpt "$CKPT" --num_steps 6 --ot_ode \
        --target_L "$Lout" \
        --num_samples 64 \
        --out_csv "$target" \
        --device cuda
done
echo "[eval_target_L] done."
