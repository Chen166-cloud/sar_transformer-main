# 可复现实验入口

以下入口不会覆盖非空结果目录，每次运行必须提供新的输出目录。每个训练/评价目录都会保存参数、源码哈希、检查点哈希、运行时版本和逐轮/逐图指标。

## 1. 真实 SAR 分组切分

```powershell
python resplit_real_sar_dataset.py --dataset_dir datasets/real_sar_dataset --seed 42
python resplit_real_sar_dataset.py --dataset_dir datasets/real_sar_dataset --verify datasets/real_sar_dataset/real_split_grouped_seed42.json
```

该命令只生成 manifest，不移动历史 train/val/test 文件。新 loader 按 manifest 中的相对路径读取。

## 2. NWPU-RESISC45 合成数据

复现最终数据时，必须保留已冻结的 v2 `generation_plan.json`（包含两次源内容去重交换）和两张分配表；不要在空目录中重新使用未修正的初始划分。已有完整数据仅运行 `--verify-only`。

```powershell
docker compose run --rm sar python generate_nwpu_sar_global_l_dataset.py --source-root datasets/NWPU-RESISC45 --output-root datasets/NWPU_RESISC45_SAR_global_L_v2 --plan-only
docker compose run --rm sar python generate_nwpu_sar_global_l_dataset.py --source-root datasets/NWPU-RESISC45 --output-root datasets/NWPU_RESISC45_SAR_global_L_v2
docker compose run --rm sar python generate_nwpu_sar_global_l_dataset.py --source-root datasets/NWPU-RESISC45 --output-root datasets/NWPU_RESISC45_SAR_global_L_v2 --verify-only
```

初始 v1 到最终 v2 的确定性修正由 `tools/repair_nwpu_global_l_split.py` 实现，并记录在 v2 manifest 的 `split.repair_swaps` 中。旧混合 L 鲁棒性对比使用 `global_L_v2_matched_split_manifest.json`。只允许 v2 进入正式训练。

主协议为单图单一全局 `L∈{1,2,4,8}`。每类 700 张按固定 seed 分层为 560 张训练源图与 140 张验证源图；训练每类每 L 严格为 140 张，总训练对数 25200。相同的 6300 张验证源图在四个 L 下分别生成固定 noisy 版本，总验证对数 25200。分配表、逐样本 seed、源/输出 SHA-256 均保存。原 4×4 混合 L 数据 `datasets/NWPU_RESISC45_SAR_intensity_v1` 保留为复杂噪声/鲁棒性消融，不再作为主协议；BSDS500 仅作历史审计。详见 `NWPU_GLOBAL_L_PROTOCOL.md`。

## 3. 监督训练与单模块消融

```powershell
python train_reproducible.py --dataset-root datasets/NWPU_RESISC45_SAR_global_L_v2 --split-manifest datasets/NWPU_RESISC45_SAR_global_L_v2/dataset_manifest.json --output-dir experiments_repro/nwpu_global_L_p0_ablation/full_seed42 --ablation full --seed 42
```

将 `--ablation` 依次替换为 `wout_frequency`、`wout_gate`、`wout_fusion`、`wout_msf`、`wout_bottleneck`；正式统计使用 42/43/44 三个 seed。完整矩阵见 `configs/ablation_matrix.json`。

完整 6×3 矩阵可在已验收的容器中顺序运行；先用 `--dry-run` 审核 36 条训练/评价命令：

```powershell
docker compose run --rm sar python run_p0_ablation_matrix.py --dry-run
docker compose run --rm sar python run_p0_ablation_matrix.py
docker compose run --rm sar python summarize_p0_ablation.py
```

矩阵运行器逐实验保存独立日志和 `matrix_status.json`，已完成目录可跳过，部分目录则显式报错，避免覆盖或混接实验。训练默认每 100 batch 写入 `batch_progress.jsonl`；每个 epoch 保存逐 L 验证 loss/PSNR/SSIM 与宏平均。

## 4. 真实 SAR 自监督适应

```powershell
python train_selfsup_reproducible.py --real-dataset-root datasets/real_sar_dataset --split-manifest datasets/real_sar_dataset/real_split_grouped_seed42.json --synthetic-dataset-root datasets/NWPU_RESISC45_SAR_global_L_v2 --synthetic-split-manifest datasets/NWPU_RESISC45_SAR_global_L_v2/dataset_manifest.json --base-checkpoint experiments_repro/nwpu_global_L_p0_ablation/full_seed42/best_model.pth --output-dir experiments_repro/nwpu_global_L_ams_seed42 --seed 42
```

默认只微调解码/融合部分，并拒绝把合成验证 PSNR 下降超过 0.5 dB 的 epoch 选为最佳检查点。

## 5. 统一评价

```powershell
python evaluate_synthetic_reproducible.py --dataset-root datasets/NWPU_RESISC45_SAR_global_L_v2 --split-manifest datasets/NWPU_RESISC45_SAR_global_L_v2/dataset_manifest.json --split val --checkpoint experiments_repro/nwpu_global_L_p0_ablation/full_seed42/best_model.pth --output-dir test_results_repro/nwpu_global_L_p0_ablation/full_seed42
python evaluate_real_reproducible.py --dataset-root datasets/real_sar_dataset --split-manifest datasets/real_sar_dataset/real_split_grouped_seed42.json --checkpoint experiments_repro/nwpu_global_L_p0_ablation/full_seed42/best_model.pth --output-dir test_results_repro/real_global_L_base_seed42 --save-images
```

正式重跑前先运行 `python smoke_test_p0.py --device cuda --input-size 256`，再使用各入口的 `--max-*-batches` 或 `--max-images` 做端到端 smoke test。历史 `train.py`、`train_selfsup.py` 和旧测试脚本保留用于解释既有产物，不再作为论文主结果入口。

全局 L 独立评价输出 `metrics_per_image.csv`、`summary_by_L.csv` 与 `summary.json`；后者包含四个 L 的统计和不加权宏平均。矩阵汇总按每个 L 及宏平均分别报告跨 seed 的 mean ± sample std。
