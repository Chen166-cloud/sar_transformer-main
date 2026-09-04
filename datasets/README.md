# 数据集统一目录

所有数据集实际存放在 `D:\research\sar_transformer-main\datasets`。每个数据集保持独立，原文件名、训练/验证/测试划分及 manifest 均保持原样。

| 子目录 | 用途 |
| --- | --- |
| `NWPU-RESISC45` | NWPU 原始光学影像，合成数据来源 |
| `NWPU_RESISC45_SAR_global_L_v2` | 当前主实验数据，使用经过修正的全局 L 划分 |
| `NWPU_RESISC45_SAR_intensity_v1` | 区域混合 L 合成数据，用于鲁棒性消融 |
| `NWPU_RESISC45_SAR_global_L_v1` | 历史审计版本；已有跨划分同内容图像问题，不用于正式训练 |
| `real_sar_dataset` | 真实 SAR 数据，用于真实域自监督适应和评价 |
| `UCMerced_LandUse_SAR_mat` | UCMerced 合成 SAR 数据，作为外部测试来源 |
| `BSR_bsds500` | BSDS500 原始图像及附带数据 |
| `bsds500_synthetic_dataset` | 历史 BSDS500 合成数据；使用时按 `official_split_manifest.json` 选择划分 |

## 使用方式

从项目根目录运行命令，新的数据路径统一写为 `datasets/<子目录名>`，例如：

```powershell
python train_reproducible.py --dataset-root datasets/NWPU_RESISC45_SAR_global_L_v2 --split-manifest datasets/NWPU_RESISC45_SAR_global_L_v2/dataset_manifest.json --output-dir experiments_repro/nwpu_global_L_p0_ablation/full_seed42 --ablation full --seed 42
```

真实 SAR 的分组划分文件为 `datasets/real_sar_dataset/real_split_grouped_seed42.json`。混合 L 鲁棒性实验使用 `datasets/NWPU_RESISC45_SAR_intensity_v1/global_L_v2_matched_split_manifest.json`。实验协议详见项目根目录的 `REPRODUCIBLE_EXPERIMENTS.md` 和 `NWPU_GLOBAL_L_PROTOCOL.md`。

## 兼容性与迁移记录

- 原项目根目录的 8 个同名数据集入口是 Windows 目录联接（junction），指向这里的实际目录，不是额外的数据副本。
- 旧脚本默认路径和历史实验记录可以继续使用；数据生成脚本保持原样，以保留 manifest 记录的生成器哈希。
- Docker Compose 将这里的目录直接挂载到容器中的旧路径，同时整个项目挂载仍提供 `/workspace/datasets/`。
- 复制或备份数据时使用本目录。Windows 联接不是可移植目录；迁移到其他计算机时，新命令使用 `datasets/` 路径，旧默认路径可通过 Docker Compose 提供。
- `organization_inventory.json` 记录整理前的文件数、逻辑字节数、文件元数据摘要及 JSON/CSV 内容哈希；`organization_verification.json` 记录移动后核对结果。逻辑字节数不是实际磁盘占用，硬链接数据可能共享存储。
- 模型、实验输出、示例代码和空的 `test_images` 目录保留在原位置。
