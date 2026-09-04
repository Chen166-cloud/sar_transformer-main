# P0 修复与正式重跑启动报告

日期：2026-09-04（Asia/Shanghai）

> 后续状态更新：本文下方的 NWPU 混合 L 训练状态已被新协议取代。用户于
> 2026-09-04 要求改为单图单全局 L；容器 `sar-nwpu-ablation-matrix` 已停止，
> 原数据与记录保留为复杂噪声/鲁棒性消融。最新主协议、记录与数据路径分别见
> `NWPU_GLOBAL_L_PROTOCOL.md`、`EXPERIMENT_RECORD_2026-09-04_GLOBAL_L.md`
> 和 `NWPU_RESISC45_SAR_global_L_v2/`。以下内容保留为此前阶段的历史记录。

## 结论

交接材料要求的 P0 工程修复、可追溯协议和 smoke test 已完成。最初误启动的
BSDS500 正式矩阵已按用户要求停止，并保留为隔离的非论文产物。正式数据已改回
中期报告指定的 NWPU-RESISC45。31,500 张合成数据已经全部生成，正式
`6 configurations × 3 seeds × 100 epochs` 矩阵由容器
`sar-nwpu-ablation-matrix` 执行。未完成前不得把 smoke test 数值或历史报告
数值写入论文主表。

## 已完成的修复

1. 真实 SAR 改为按 512×512 父图分组的只读 manifest，不再移动历史数据。
   新划分包含 5,876 个 patch、1,469 个父图；train/val/test 分别为
   4,700/584/592 个 patch、1,175/146/148 个父图，三组父图交集均为 0。
2. 建立 `intensity_v1` 数值域：合成数据不再开平方；真实数据统一使用逐图
   1%/99% 分位归一化；log 分支在融合前使用精确逆变换回强度域。
3. 建立 Full、w/o FFT、w/o Gate、w/o Fusion、w/o MSF、w/o Bottle 六个
   单变量配置。同 seed 下先完整同序初始化再移除目标模块；所有共享参数逐张量
   哈希一致。
4. PSNR/SSIM 固定 `data_range=1`；真实 SAR 的 ENL/M/EPI 共用 noisy 图上一次
   选定的 ROI，并输出逐图记录、均值、中位数及 bootstrap 95% CI。
5. BSDS500 历史 `train/` 目录被确认混入 200 张 `tst_*`。新增官方来源 manifest，
   按文件名前缀恢复 train/val/test=200/100/200；500 个文件 SHA-256 全部匹配，
   重复、漏分和意外文件均为 0。
6. 新训练与评测入口拒绝覆盖非空结果目录，保存运行配置、源码/manifest/检查点
   SHA-256、运行时、逐 epoch/逐图指标和 best/last 检查点。训练 CSV 每个 epoch
   即时追加，避免长任务中断后丢失已有日志。
7. AMS 默认仅微调解码/融合部分，并用固定合成验证集限制遗忘：合成 PSNR 下降
   超过 0.5 dB 的 epoch 不得成为最佳检查点。

## 验收结果

- Python 静态编译：通过。
- 协议单元测试：13/13 通过。
- Docker 环境：Python 3.10.14、PyTorch 2.2.2、CUDA 12.1、RTX 4070 Laptop GPU。
- 历史 Base 检查点：`strict=True` 完整加载，missing/unexpected keys 均为空。
- 256×256 GPU 前向：输入输出形状一致，结果全部有限。
- 六个消融配置 64×64 前向：全部有限；共享初始化 mismatch 均为 0。
- 256×256 GPU 反向传播：通过，无显存溢出。
- 官方 manifest 端到端 smoke：监督训练、独立 test 评测、真实 AMS 微调均通过。
- 真实 SAR manifest：5,876 个文件完整覆盖，重复/缺失/group mismatch 均为 0。

smoke test 只执行少量 batch，用于验证代码路径，数值不构成论文结果。

## 已停止的错误任务

- 容器名：`sar-p0-ablation-matrix`（已停止，未删除）
- 启动容器 ID：`343c6942ac3320e484d88c07b23d48a4c575fff522fabd542b620d21f1aaa642`
- 停止时任务：BSDS500 `full_seed42`，已完成 45/100 epochs。
- 状态文件：`experiments_repro/p0_ablation/matrix_status.json`
- 逐任务日志：`experiments_repro/p0_ablation/_runner_logs/`
- 训练产物：`experiments_repro/p0_ablation/<configuration>_seed<seed>/`
- 测试产物：`test_results_repro/p0_ablation/<configuration>_seed<seed>/`

这些文件不得作为中期报告的 NWPU 主实验结果。

## NWPU 正式流水线

- 原图：45 类 × 700 张，共 31,500 张，已完整核验。
- 划分：每类 train=560、val=140，总计 25,200/6,300。
- 合成协议：严格对应报告表 6 的综合 SAR 噪声参数。
- 合成状态：31,500/31,500 完成，共约 15.388 GiB，无临时 `.mat` 文件。
- 正式训练容器：`sar-nwpu-ablation-matrix`，容器 ID
  `dba3b3c6f4b7786b42aea701195e6e408f90b701c3b36220c79afbce6b489ec1`。
- 当前任务：NWPU `full_seed42`，100 epochs，256×256，正在运行。
- 合成数据：`NWPU_RESISC45_SAR_intensity_v1/`。
- 正式训练：`experiments_repro/nwpu_p0_ablation/`。
- 正式验证：`test_results_repro/nwpu_p0_ablation/`。

矩阵完成后运行：

```powershell
docker compose run --rm sar python summarize_p0_ablation.py
```

该命令只有在 18 个 test summary 全部存在时才生成 mean ± sample std 汇总，否则
会明确列出缺失任务。

## 后续证据边界

本次完成的是交接要求的 P0 修复与正式重跑启动。P1 的 UCM 固定外测、Lee、
SAR-BM3D、公开深度/真实自监督基线及最终排版图仍需外部数据、实现或权重，不能
用现有历史数字替代。在正式矩阵和这些基线完成前，不应生成带未经验证结论的论文
主表或摘要。
