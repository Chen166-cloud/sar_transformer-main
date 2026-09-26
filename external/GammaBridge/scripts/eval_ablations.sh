#!/usr/bin/env bash
# Wait for both ablation ckpts then run NFE=1/5/25 eval on each.
# Usage: bash scripts/eval_ablations.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

CKPT_A="weights/abl_linear_sched/ckpt_0015000.pt"
CKPT_B="weights/abl_direct_pred/ckpt_0015000.pt"
RESULTS="eval/results"

echo "[eval_ablations] waiting for checkpoints..."
for ckpt in "$CKPT_A" "$CKPT_B"; do
    while [ ! -f "$ckpt" ]; do
        echo "  not ready: $ckpt  ($(date '+%H:%M:%S'))"
        sleep 60
    done
    echo "  found: $ckpt"
done

for nfe in 2 6 26; do
    case $nfe in
        2)  tag="ns2"  ;;
        6)  tag="ns6"  ;;
        26) tag="ns26" ;;
    esac

    echo "[eval] abl_linear_sched NFE=$nfe ..."
    python eval/run_eval.py \
        --ckpt "$CKPT_A" --num_steps "$nfe" --ot_ode \
        --num_samples 64 \
        --out_csv "$RESULTS/abl_linear_sched_${tag}.csv" \
        --device cuda

    echo "[eval] abl_direct_pred NFE=$nfe ..."
    CUDA_VISIBLE_DEVICES=1 python eval/run_eval.py \
        --ckpt "$CKPT_B" --num_steps "$nfe" --ot_ode \
        --num_samples 64 \
        --out_csv "$RESULTS/abl_direct_pred_${tag}.csv" \
        --device cuda
done

echo "[eval_ablations] done."
