#!/usr/bin/env bash
# Finer NFE sweep on all 5 models (full + 4 ablations).
# Existing: ns2 (NFE=1), ns6 (NFE=5), ns26 (NFE=25).
# Adding:   ns3 (NFE=2), ns4 (NFE=3), ns11 (NFE=10), ns16 (NFE=15).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

declare -A CKPTS=(
    [combo_v1_L1]="weights/combo_v1_L1_ckpt_0015000.pt"
    [abl_naive_post]="weights/abl_naive_post/ckpt_0015000.pt"
    [abl_no_cond]="weights/abl_no_cond/ckpt_0015000.pt"
    [abl_linear_sched]="weights/abl_linear_sched/ckpt_0015000.pt"
    [abl_direct_pred]="weights/abl_direct_pred/ckpt_0015000.pt"
)

for name in "${!CKPTS[@]}"; do
    ckpt=${CKPTS[$name]}
    if [ ! -f "$ckpt" ]; then echo "MISS $ckpt"; continue; fi
    outdir="eval/results"
    [ "$name" == "combo_v1_L1" ] && outdir="eval/results/combo_v1_L1" && stem="ckpt0015000" || stem="$name"
    for nfe in 3 4 11 16; do
        target="$outdir/${stem}_ns${nfe}.csv"
        if [ -f "$target" ]; then echo "SKIP $target (exists)"; continue; fi
        echo "[$name] NFE=$((nfe-1))"
        CUDA_VISIBLE_DEVICES=0 python3 eval/run_eval.py \
            --ckpt "$ckpt" --num_steps "$nfe" --ot_ode \
            --num_samples 64 \
            --out_csv "$target" \
            --device cuda
    done
done
echo "[eval_finer_nfe] done."
