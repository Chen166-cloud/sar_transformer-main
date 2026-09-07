# ICSPS 2026 SAR 去斑正式实验记录表

> 协议：`ICSPS26-FROZEN-v2`  
> 论文范围：SAR 去斑；不包含舰船检测。  
> 使用方法：脚本生成的 CSV/JSON 是原始证据，本文件只用于人工核对和填写论文表格。不得把 smoke、历史 compound-noise 或中期报告数字填入正式结果。

> **顺序警告：** `data_audit.csv`、`validation_selection.csv`、`figure_selection.csv` 必须在 test 解锁前完成；尤其后两者会被一次性 seal 绑定，不能等所有实验结束后补填。其余主表只从已验证的 CSV/JSON 抄录，不从终端显示或 TensorBoard 曲线目测取数。完整逐表字段来源见远程运行手册第 9 节。

## 0. 冻结信息

| 项目 | 记录值 |
|---|---|
| 仓库 commit |  |
| 代码 bundle SHA-256 |  |
| GPU 0 型号 / 实测显存 |  |
| GPU 1 型号 / 实测显存 |  |
| CPU 型号 / 可用 vCPU |  |
| 实测 RAM |  |
| 数据盘挂载点 / 总量 / 运行前可用量 |  |
| Docker image ID 或 Conda lock |  |
| NVIDIA driver / `nvidia-smi` CUDA capability |  |
| Python / PyTorch / `torch.version.cuda` |  |
| 正式协议配置 SHA-256 |  |
| NWPU source manifest SHA-256 |  |
| seed 42 schedule SHA-256 |  |
| seed 43 schedule SHA-256 |  |
| seed 44 schedule SHA-256 |  |
| UCM fixed-pair manifest SHA-256 |  |
| Real QC manifest SHA-256 |  |
| Pretest seal SHA-256 |  |
| test 解锁时间（UTC） |  |

如果改变数据划分、噪声模型、指标、更新预算或模型选择规则，停止填写并在 `deviation_log.csv` 中登记；受影响实验统一升版并重跑。

## 1. 数据准备验收

| 检查 | NWPU train | NWPU val | UCM test | Real train / val / test | 通过？ |
|---|---:|---:|---:|---:|:---:|
| 独立 source / child 数 | 900 | 225 | 2,100 | 4,700 / 584 / 592 assigned |  |
| parent 数 | – | – | – | 1,175 / 146 / 148 |  |
| fixed pairs | online 100,000 | 900 | 8,400 | – |  |
| 每个 $L=1,2,4,8$ 数量 | 各 25,000 | 各 225 | 各 2,100 | – |  |
| missing / duplicate ID | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |  |
| 跨 split path/content overlap | 0 / 0 | 0 / 0 | 与 NWPU content overlap=0 | parent overlap=0 |  |
| 非法、constant 或未配对排除数 | – | – | – |  /  /  |  |
| manifest audit 状态 |  |  |  |  |  |

Gamma clipping 后不再严格保持单位均值，因此下面数值必须由生成脚本实测填写。
各 `saturation` 格统一填对应 L 的 `saturation_rate_mean`；`preclip max 范围` 填四个 L 的整体 min–max，各 L 的详细 min/max 保留在 JSON 或 `data_audit.csv` notes，不凭目测填写。

| Dataset | $L=1$ saturation | $L=2$ | $L=4$ | $L=8$ | preclip max 范围 | 报告来源 |
|---|---:|---:|---:|---:|---:|---|
| NWPU train schedule（全 100k updates 累计） |  |  |  |  |  | 各正式 run `completion.json.training_synthesis_by_L` |
| NWPU fixed validation |  |  |  |  |  |  |
| UCM fixed test |  |  |  |  |  |  |

## 2. Smoke test（不得用于论文）

| 项目 | 预期 | 实测 | 通过？ | 日志 |
|---|---|---|:---:|---|
| 模型 forward：TransSARV2 | `1×1×256×256 → 1×1×256×256` |  |  |  |
| 模型 forward：Full / 三个核心消融 | 同上 |  |  |  |
| SAR-CAM pinned commit | `ea5ee3b…` 且 clean checkout |  |  |  |
| 48-row smoke schedule 可重放 | 相同 seed 输出 hash 相同 |  |  |  |
| 中断续训 | 下一行为 `global_step+1` |  |  |  |
| checkpoint 错配拒绝 | method/variant/hash 错误时 fail |  |  |  |
| synthetic evaluator | 输出 per-image CSV + aggregate JSON |  |  |  |
| real evaluator | 所有方法使用相同 valid IDs / ROI |  |  |  |

## 3. 统一监督训练记录

正式核心顺序建议：先 `Full(42)` 和 `TransSARV2(42)`，再三个消融；资源允许时补 seed 43/44 和 SAR-CAM。每个可学习方法均为 100,000 successful updates，step 0 与每 5,000 updates 验证。

| Run ID | Method | Variant | Seed | Schedule hash | Best step | Best val macro MSE | Last step | Wall time | Best checkpoint hash | 状态/备注 |
|---|---|---|---:|---|---:|---:|---:|---:|---|---|
|  | Ours | full | 42 |  |  |  |  |  |  |  |
|  | TransSARV2 | – | 42 |  |  |  |  |  |  |  |
|  | Ours | intensity_only | 42 |  |  |  |  |  |  |  |
|  | Ours | log_only | 42 |  |  |  |  |  |  |  |  |
|  | Ours | wout_all_fdr | 42 |  |  |  |  |  |  |  |  |
|  | SAR-CAM | – | 42 |  |  |  |  |  |  |  |  |
|  | Ours | full | 43 |  |  |  |  |  |  |  |
|  | Ours | full | 44 |  |  |  |  |  |  |  |
|  | TransSARV2 | – | 43 |  |  |  |  |  |  |  |
|  | TransSARV2 | – | 44 |  |  |  |  |  |  |  |

## 4. UCM-21 外部合成主表

不得在 UCM 上调 checkpoint、Lee window、模型或样例。Macro 是 21 类与四个 $L$ 等权结果，不是简单把 8,400 行当 IID 样本。

| Method | Mode | $L=1$ PSNR / SSIM | $L=2$ | $L=4$ | $L=8$ | Macro PSNR / SSIM |
|---|---|---:|---:|---:|---:|---:|
| Noisy | – |  |  |  |  |  |
| Lee | oracle-$L$ |  |  |  |  |  |
| SAR-BM3D | oracle-$L$ |  |  |  |  |  |
| SAR-CAM | blind, retrained |  |  |  |  |  |
| TransSARV2 | blind, retrained |  |  |  |  |  |
| Ours | blind, one model |  |  |  |  |  |

| Paired comparison | Metric direction | Difference | Source-cluster 95% CI | Bootstrap draws | 结论 |
|---|---|---:|---:|---:|---|
| Ours − strongest non-ours | PSNR ↑ |  | [ , ] | 10,000 |  |
| Ours − strongest non-ours | SSIM ↑ |  | [ , ] | 10,000 |  |

上表是六页论文的 paper-facing primary subset；完整的 3 组预注册比较 × PSNR/SSIM（共 6 行）必须全部填入 `ucm_paired_comparisons.csv` 并保留。

## 5. 核心消融表

| Variant | Log / inverse | Compensation | Bottleneck / decoder FDR | Macro PSNR / SSIM |
|---|:---:|:---:|:---:|---:|
| Intensity-only | × / – | × | ✓ / ✓ |  |
| Log-only | ✓ / ✓ | × | ✓ / ✓ |  |
| Full w/o all FDR | ✓ / ✓ | ✓ | × / × |  |
| Full | ✓ / ✓ | ✓ | ✓ / ✓ |  |

## 6. AMS 记录

AMS 只使用真实 SAR train/validation，只按固定真实 validation masked loss 选择 epoch，并只在相同的真实 SAR test IDs 上比较适应前后结果。

| Base checkpoint | Seed | Epoch | Fixed real-val masked loss | Selected |
|---|---:|---:|---:|:---:|
|  | 42 | 0 | – |  |
|  | 42 | 1 |  |  |
|  | 42 | 2 |  |  |
|  | 42 | 3 |  |  |
|  | 42 | 4 |  |  |
|  | 42 | 5 |  |  |
|  | 42 | 6 |  |  |
|  | 42 | 7 |  |  |
|  | 42 | 8 |  |  |

## 7. 真实 SAR 主表

本论文已在 test 解锁前冻结为不使用 GT16，因此不报告任何 GT16-based quasi-reference metric，也不在主表中保留这些空列。真实 SAR 按用户指定的中期报告指标记录 ENL、M 和 EPI；所有方法必须使用同一实现、同一测试 patch 和同一固定 ROI。

| Method | Adaptation | Test patches / parents | ENL | M ↓ | EPI ↑ |
|---|---|---:|---:|---:|---:|
| Noisy | – |  |  | N/A$^*$ | 1.0000$^*$ |
| TransSARV2 | synthetic only |  |  |  |  |
| Ours-base | synthetic only |  |  |  |  |
| Ours+AMS | encoder frozen |  |  |  |  |

$^*$ 对 Noisy identity pass-through，$\widehat X=Y$；M 在严格公式下无有效窗口，记为 N/A；EPI=1 是恒等关系，只作输入锚点，不参与优劣排名。

## 8. 不纳入本文的测量

本次论文不测量、不填写 Params、MACs、latency 或 peak memory。

## 9. 图片与样例冻结记录

| 图 | Source / parent ID | 类别与 $L$ | 固定 crop | 选择时间 | 选择规则 | 输出路径 |
|---|---|---|---|---|---|---|
| 图 3：PSNR/SSIM 跨视数曲线 | 全 UCM | 全 21 类、四个 $L$ | – |  | 全量统计 |  |
| 图 4：UCM 合成定性对比 |  |  |  | test 前 | 预注册细线/规则纹理/亮区 |  |
| 图 5：真实 SAR + ratio + AMS |  | homogeneous + structured |  | test 前 | 预注册 parent/crop |  |

## 10. 最终签核

- [ ] 所有正式模型使用相同 source manifest、同 seed schedule、100k updates 和选模规则。
- [ ] 每张论文表都能由保存的 per-image/per-patch CSV 重建。
- [ ] checkpoint 包含 optimizer、scheduler、global step、RNG、protocol hashes，并 strict-load 通过。
- [ ] UCM bootstrap 单位为 source，real bootstrap 单位为 parent。
- [ ] 单 seed 结果没有写成 training-run mean±std。
- [ ] 所有图片样例/ROI 在 test 解锁前冻结，显示 stretch 未用于指标。
- [ ] `pretest_seal.json` 在任何 test 输出前生成；所有正式 aggregate/汇总/图片的 seal hash 一致。
- [ ] M 与 EPI 已按各自所引原论文公式核对；没有把内部诊断量改名写入主表。
- [ ] 历史数据、smoke 数字和舰船检测内容均未进入论文。
- [ ] 运行失败、排除项和协议偏差均有原始日志。

机器可读空表位于 `records/icsps2026_templates/`。`ucm_main_table.csv` 的 PSNR/SSIM 取自 `ucm_main_summary.csv`；`real_main_table.csv` 只记录预注册的真实 SAR 指标。no-GT16 决策已落实为删除全部 q 指标行。两个 `*_paired_comparisons.csv` 按相应汇总脚本输出逐项填写；若 test 前冻结的 primary comparator 是 SAR-CAM，将 UCM paired 模板中的 `baseline_key=transsar` 改成 `sar_cam`。三 seed 的均值与样本标准差保留在 `ucm_multiseed_summary.csv`，再抄入论文显示列，不要硬塞进单 seed 主表。建议复制一份到服务器的实验输出根目录，保留模板原件不改。
