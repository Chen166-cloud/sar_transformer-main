# SAR 数值域契约（intensity_v1）

状态：论文重跑的强制协议。历史结果仅可通过显式 `legacy_amplitude_v0` 复现，不能与新结果混表。

## 统一定义

- 物理/数值语义：所有训练目标、网络最终输出、保存的数值结果和评价指标均为归一化强度（intensity）。
- 有效范围：`[0,1]`，`float32`；含 NaN/Inf 或不可归一化的常量真实图像直接报错/跳过并记录。
- 合成 `.mat`：`clean` 与 `noisy` 均视为强度；从磁盘读取后裁剪到 `[0,1]`，不再开平方。
- 真实 `.mat`：对每幅图使用同一个 1%/99% 分位裁剪，再线性映射到 `[0,1]`。AMS 训练、验证和真实测试调用同一实现。
- 可视化：强度图默认线性显示；如需幅度显示，只允许在展示副本上执行 `sqrt`，不得用该副本计算指标。

## 模型变换链

`intensity input x -> clamp[0,1] -> z=log1p(10x)/log(11) -> log branch -> sigmoid -> x_log=(exp(z_hat*log(11))-1)/10 -> optional original-intensity residual fusion -> clamp[0,1]`

`log_transform_01` 与逆变换由 `numeric_domain.py` 唯一实现。单元测试要求 `[0,1]` 上往返最大绝对误差不超过 `1e-6`。

## 各环节

| 环节 | 磁盘输入 | 网络输入/目标 | 网络输出 | 指标域 |
|---|---|---|---|---|
| BSDS/NWPU 监督训练 | intensity clean/noisy | intensity/intensity | intensity | intensity |
| UCM 外测 | intensity clean/noisy | intensity | intensity | intensity，`data_range=1` |
| 真实 SAR AMS | 原始强度数组 | percentile-normalized intensity | intensity | 真实验证仅作稳定性监控 |
| 真实 SAR 测试 | 原始强度数组 | 与 AMS 相同的 percentile-normalized intensity | intensity | intensity |

## 指标协议

- PSNR：逐图计算，固定 `data_range=1`，默认不裁边。
- SSIM：逐图 11x11 Gaussian 窗（sigma=1.5），固定 `data_range=1`，使用有效窗口区域，默认不裁边。
- ENL：只在 noisy 输入上选择低方差 ROI；同一坐标同时用于 noisy、Base、AMS 和所有基线，禁止在各输出上重新选 ROI。
- M：在正值下限 `1e-6` 的 intensity ratio map 上计算；随机置乱固定 seed 并记录。
- EPI：在同一 intensity 图对上计算 Sobel 梯度相关性。EPI 必须与 ENL、M、ratio/edge map 联合解释，不能单独证明去斑质量。

## 版本隔离

- `intensity_v1` 是所有新论文实验默认值。
- `legacy_amplitude_v0` 仅复现旧检查点：合成数据开平方、旧 log 分支不执行逆变换。任何产物必须在配置中带出该名称。
- 不同数值域的检查点、CSV 和表格不得合并求平均或直接比较。
