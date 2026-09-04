# NWPU-RESISC45 全局 L 主实验协议

## 数据角色

- 主实验：`datasets/NWPU_RESISC45_SAR_global_L_v2`。初版 v1 因发现两组跨划分同内容图像，已降级为审计中间产物，不得训练。
- 鲁棒性消融：保留原 `datasets/NWPU_RESISC45_SAR_intensity_v1`，其中采用 4×4 区域混合 L；不得与主实验指标混报。
- 外部测试：UCMerced_LandUse，不参与 NWPU 训练或验证划分。

## 可复现划分

- NWPU-RESISC45 共 45 类，每类 700 张。
- 使用 `split_seed=42` 对每类独立、确定性地划分 560 张训练源图和 140 张验证源图。
- 训练与验证源路径交集、源文件内容 SHA-256 交集均必须为 0。
- 初始划分发现 airport 类中两组同内容异名图跨集合；按固定种子做两次同类别交换，保持每类 560/140 和每类每 L=140 不变。交换详情写入 `split.repair_swaps`。
- 混合 L 鲁棒性数据使用 `datasets/NWPU_RESISC45_SAR_intensity_v1/global_L_v2_matched_split_manifest.json`，与修正后的主实验 clean 划分一致。原始混合 L 文件及原 manifest 不改动。

## 训练集 L 分配

- 每张训练图像只使用一个全局 `L`。
- 候选值为 `{1,2,4,8}`，`assignment_seed=42`。
- 每个类别在每个 L 下严格分配 140 张，因此每个 L 共 6300 张训练图，总训练对数为 25200。
- 完整分配保存在 `train_l_assignment.csv`；每个输出文件的样本随机种子同时写入数据 manifest。

## 验证集与汇总

- 同一批 6300 张验证 clean 源图分别生成 L1、L2、L4、L8 四套固定 noisy 版本。
- 每套 6300 对，总验证对数为 25200。
- 分别报告每个 L 的 noisy/prediction PSNR、SSIM，并报告不加权宏平均：

  `Macro = (Metric_L1 + Metric_L2 + Metric_L4 + Metric_L8) / 4`

- 验证样本、L、输出路径与随机种子均固定，不在 epoch 之间重新采样。

## 合成模型

- 强度域：`clean = normalized_grayscale^2`，像素范围 `[0,1]`。
- 全图使用 `N ~ Gamma(shape=L, scale=1/L)`，`noisy = clean * N`。
- 强散射掩膜仅控制增益与稀疏脉冲，不再把局部 L 强制改为 1，因而整幅图始终只有一个 L。
- 其他复合退化参数沿用报告设置，并在四个 L 下保持同一协议：加性噪声 `sigma=0.015`、错位混合 `alpha=0.08`、位移 `(1,2)`、条纹振幅 `0.02`、行列概率各 `0.08`。

## 记录与验收

- 生成器：`generate_nwpu_sar_global_l_dataset.py`。
- 协议配置：`configs/nwpu_global_l_protocol.json`。
- 数据根目录必须包含 `generation_plan.json`、`train_l_assignment.csv`、`validation_l_sets.csv` 与 `dataset_manifest.json`。
- manifest 验证必须通过：训练每类每 L=140、验证每 L=6300、训练/验证源图交集=0、所有 `.mat` 路径唯一且完整。
- 同内容泄漏修正由 `tools/repair_nwpu_global_l_split.py` 可复现；50390 个未变化样本以硬链接复用，仅重新生成10对，避免重复占用整套数据空间。所有版本必须保持只读，不得原位修改共享 MAT 文件。
- 完整验收入口：`tools/verify_nwpu_global_l_artifacts.py`，包括最终50400个文件的SHA-256复核。
- 训练写入 `batch_progress.jsonl`，默认每 100 batch 记录一次；epoch 结果写入 `epoch_metrics.csv`，包含逐 L 指标与宏平均。
