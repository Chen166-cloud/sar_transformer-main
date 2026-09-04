# 2026-09-04 NWPU 全局 L 协议切换记录

## 用户确认的方案

1. 每张训练图仅使用一个全局 L，候选 `{1,2,4,8}`，按类别均衡分配。
2. 保存训练 L 分配表及随机种子。
3. 验证集建立 L1、L2、L4、L8 四套固定版本。
4. 分别报告各 L 的 PSNR/SSIM，再报告不加权宏平均。
5. 保留原 4×4 区域混合 L 数据作为复杂噪声/鲁棒性消融，不作为唯一主实验。

## 已执行

- 已停止旧容器 `sar-nwpu-ablation-matrix`；停止详情见 `experiments_repro/nwpu_p0_ablation/STOPPED_PROTOCOL_CHANGE.md`。
- 新生成器：`generate_nwpu_sar_global_l_dataset.py`。
- 新数据目录：`NWPU_RESISC45_SAR_global_L_v2`，未覆盖旧数据。
- `split_seed=42`，`assignment_seed=42`，`synthesis_seed=20260904`。
- 已核验训练 25200 对、验证 25200 对（四套各 6300），训练/验证唯一 clean 源图分别为 25200/6300。
- 训练每类每 L 均为 140；全训练集每 L 均为 6300。
- 训练/验证源路径及内容 SHA-256 交集均为 0；鲁棒性数据使用新建的 `global_L_v2_matched_split_manifest.json` 与最终主实验划分匹配。
- 18 项协议/数值域/指标/消融单元测试通过。
- 训练器新增逐 L 验证指标、宏平均及 `batch_progress.jsonl`；评价器新增 `summary_by_L.csv` 和 JSON 宏平均。
- 修复旧数据加载器的 PIL 8 位量化，改为保留 float32；GPU 端到端 smoke v2 中，训练验证与独立评价的四个 L 及宏平均 PSNR/SSIM 完全一致。
- smoke v2 记录位于 `experiments_smoke/nwpu_global_L_full_256_seed42_v2` 与 `test_results_smoke/nwpu_global_L_full_256_seed42_v2`；数值不作为论文结果。
- 主实验保留报告中的增益、稀疏脉冲、加性热噪声、错位与条纹参数；强散射掩膜不再改变 L，保证每张图只有一个全局 L。

## 报告

- 原件保留：`codex_handoff_2026-09-03/source_materials/陈虹羽中期报告.docx`。
- 修订版：`codex_handoff_2026-09-03/source_materials/陈虹羽中期报告_全局L协议修订版_v2.docx`。
- 已更新摘要、数据与噪声协议、训练验证流程、逐 L/宏平均结果模板、历史结果标记与后续计划。
- 当前环境缺少 LibreOffice/soffice，标准渲染失败；已执行结构检查，未声称通过视觉版式验收。

## 最终数据验收与训练状态

- 新数据50400对已生成；初版发现两组 airport 同内容异名图跨划分，已做两次确定性同类别交换。v2通过硬链接复用50390对，重新生成10对，配额不变。v1仅作审计中间产物，未用于正式训练。
- 最终v2完整文件哈希验收已通过：50400/50400，缺失/重复/未列入文件均为0；源路径和源内容SHA-256跨划分交集均为0。机器可读结果为 `NWPU_RESISC45_SAR_global_L_v2/verification_report.json`。
- 最终 manifest SHA-256：`748b7010bf4f4c29eac5c814e32ba0f360e49ae7e063a20e03981a231c75e9bb`。
- 实际完整manifest的256×256 GPU训练/四L验证 smoke test 已通过，记录位于 `experiments_smoke/nwpu_global_L_v2_final_manifest`。此前小清单 smoke v2 已证明训练验证与独立评价逐L及宏平均完全一致。
- 最后一次完整manifest独立评价复测未完成：Docker引擎返回500/502，随后发现WSL后端停止。正式新矩阵尚未启动，不存在新协议正式epoch结果。
- 已修复独立评价中逐文件真实路径解析的多余开销；恢复Docker后可继续评价 smoke，再启动矩阵。

## Docker 恢复阻塞（2026-09-04）

- 正常 `docker desktop restart --detach` 未恢复引擎。后端日志明确报错：`initializing Inference manager ... dockerInference ... The file cannot be accessed by the system`。
- 与交接材料相同的两个失效运行socket：`C:\Users\Chen\AppData\Local\Docker\run\dockerInference` 和 `C:\Users\Chen\AppData\Local\Docker\run\userAnalyticsOtlpHttp.sock`，均为0字节重解析点。
- 在关闭故障后端后尝试重命名保留，系统拒绝访问；随后仅针对这两个socket的清理命令被执行环境策略拦截，未执行，未删除任何文件。
- 需要用户恢复Docker Desktop引擎后再继续；不得恢复出厂设置或删除镜像、容器、VHDX、数据集、检查点。
- C盘观测剩余约1.1 GB，仅作为容量风险记录，不认定为本次故障原因。

后续实验候选顺序（暂不执行）：先以 `test_results_smoke/nwpu_global_L_v2_final_manifest_fastpaths` 为新输出目录完成独立评价复测；核对训练与独立评价逐L/宏平均相等；再用 `run_p0_ablation_matrix.py` 默认参数启动正式矩阵。默认数据源为全局L v2，输出根为 `experiments_repro/nwpu_global_L_p0_ablation`，不加载旧权重，每100 batch写入进度。此段仅保留技术计划，不构成恢复后自动启动的授权。

## 本轮 Docker 修复尝试与暂停要求（2026-09-04 11:50，UTC+08:00）

- 用户最新要求：只修复 Docker Desktop 引擎；修复完成后告知，先不要开始实验。该要求覆盖前述恢复后继续实验的旧说明。训练、评价和 smoke test 均须等待用户新的明确授权。
- 本轮开始时未发现 Docker/vmmem 进程，两个失效 socket 仍为 0 字节重解析点。
- 按用户明确授权再次尝试仅用 `Remove-Item -LiteralPath` 清理上述两个临时 socket，命令在创建进程前被环境策略拒绝（`blocked by policy`），未执行，未删除文件；未改用其他方式绕过限制。
- 对两个文件执行只读 `fsutil reparsepoint query` 均返回 Windows 错误 1920（系统无法访问此文件）。
- 通过官方 Docker Desktop 可执行程序正常启动后，11:49:13 的后端日志再次报 `initializing Inference manager ... dockerInference ... The file cannot be accessed by the system`；引擎未恢复。
- 11:49:17 日志出现 GUI 的 `Reset to factory defaults` 点击及 `resetting the application`。用户随后确认是其手动点击；本代理未发起或追加重置。11:50 检查时未见重置完成日志。
- `docker_data.vhdx` 仍存在，大小 13994295296 字节，最后修改时间为 02:51:36；`ext4.vhdx` 仍存在，大小 130023424 字节，最后修改时间为 02:56:35。文件存在不能代替镜像/容器完整性验收，须待引擎恢复后只读检查。
- `docker info --format` 长时间无响应，已只终止本次健康检查的 CLI 进程；未结束或接管用户点击的重置流程。尚不能宣称修复成功。
- 当前 C 盘可用约 0.35 GB，D 盘约 36.15 GB。容量不足是额外风险，未擅自清理其他文件。
- 未发现宿主 Python 训练进程；只读检查未发现项目脚本的 Docker restart policy、相关 Windows 计划任务或 Codex 自动任务。由于引擎不可用，尚无法完成 `docker ps`/容器 restart policy 核验。
- 本轮未启动任何训练、评价或 smoke test，未修改数据集、检查点、镜像或报告文件。后续仅可继续引擎修复和只读健康检查。
