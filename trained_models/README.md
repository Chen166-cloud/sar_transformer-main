# 已训练模型汇总

本目录集中保存项目中具有代表性的已训练权重。整理日期：2026-09-07。

## 收录范围

- 6 种网络结构的历史完整监督训练最佳权重。
- `TransSARV2_DualFreqNG_Bottle` 的 AMS 真实 SAR 自监督适应最佳权重。
- 上游预训练基线和项目根目录遗留基线各一份。
- 共 9 个权重文件；均为源文件的完整副本，SHA-256 已逐个核对一致。
- 原始权重仍保留在原位置，没有移动或删除。

逐文件的结构、轮次、指标、参数量、原始位置和 SHA-256 见 `MANIFEST.csv`。

## 使用建议

- 当前规范架构（DFNG-SARNet）的历史监督权重：`TransSARV2_DualFreqNG_Bottle_best_e47.pth`。
- 面向真实 SAR 的 AMS 适应权重：`TransSARV2_DualFreqNG_Bottle_AMS_best_e8.pth`。
- 复现上游 `TransSARV2` 示例：`TransSARV2_upstream_pretrained.pth`。该文件的参数名带 `module.` 前缀，加载时需要兼容 DataParallel 格式。
- 历史验证记录中，`TransSARV2_FreqNG_best_e50.pth` 的验证损失最低、SSIM 最高；`TransSARV2_Full_best_e40.pth` 的 PSNR 最高。

## 未收录

- `experiments/*/all/` 下的逐轮快照。
- 只训练到 45/100 epoch 的 `experiments_repro/p0_ablation/full_seed42`。
- `experiments_smoke` 下的一轮或少量 batch 流程验证权重。
- 所有 `*.pth.tmp` 临时或可能不完整文件。

注意：这里的完整模型属于历史训练协议。最新 NWPU global-L v2 正式矩阵尚未产生正式 epoch 权重，不应把本目录中的历史指标当作新协议结果。
