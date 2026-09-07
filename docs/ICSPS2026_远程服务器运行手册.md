# ICSPS 2026 SAR 去斑远程服务器运行手册

> 适用协议：`ICSPS26-FROZEN-v2`  
> 适用机器：Ubuntu 22.04，Python 3.12，PyTorch 2.5.1，PyTorch CUDA build 12.4，RTX 4090 24 GiB × 2，16 vCPU，120 GB RAM  
> 论文范围：单极化、单幅、强度域 SAR 去斑；**不包含舰船检测**  
> 本手册中的 smoke/profile 输出均不得写入论文，只有未带 `--nonformal-smoke` 且完整通过校验的输出才是正式证据。

---

## 0. 实际执行顺序（先看这里）

后续章节同时承担参考手册的作用，因此章号不等于第一次部署的时间顺序。新服务器严格按下列顺序执行：

1. 运行第 3 节开头的原始硬件/PyTorch 核验命令，再用第 4.2 节确认数据盘挂载点。
2. 按第 4.2 节先补齐系统传输/解包工具并创建远端目录，再按第 4.3–4.5 和第 5.1 节打包、传输并解包代码/manifest/原始数据。
3. 按第 5.2 节配置并 `source` 环境文件，再运行可归档 inventory、依赖安装和 preflight。
4. 按第 4.6 节下载 UCM；本论文已冻结为不使用 GT16，然后从第 6.1 节开始执行 P0→P1→P2。
5. 首次 UCM/real test 前必须通过第 6.5.1 的预注册解锁门；不可先看 test 数字再填对手或图片 ID。

---

## 1. 已锁定的实验口径

| 项目 | 正式值 |
|---|---|
| Protocol ID | `ICSPS26-FROZEN-v2` |
| NWPU 监督训练源 | 900（45 类 × 20） |
| NWPU 验证源 | 225（45 类 × 5） |
| 训练预算 | 每个可学习方法 100,000 successful optimizer updates |
| Batch size | 每个进程、每张 GPU 均为 1 |
| 合成 speckle | `Gamma(shape=L, scale=1/L)`，`L∈{1,2,4,8}`；除 clip 外无附加退化 |
| 训练采样 | 冻结的 100,000 行 schedule；每个 L 25,000 次；D4 各 12,500 次 |
| 固定验证 | 225 × 4 = 900 pairs；step 0 和每 5,000 updates 验证 |
| 外部合成测试 | 完整 UCM-21：2,100 sources × 4 L = 8,400 pairs |
| 真 SAR | 1,469 个 parent，按 parent 隔离为 1,175/146/148；4,700/584/592 patches |
| 主模型选择 | NWPU validation 的 class×L macro MSE 最低；并列取更早 step |
| 核心模型 | Ours Full、TransSARV2、Intensity-only、Log-only、Full w/o all FDR |
| 传统基线 | Noisy、Lee-MMSE、官方 SAR-BM3D v1.0 |
| 可选加分基线 | 官方 SAR-CAM，同数据、同 schedule、同 100k updates 重训 |
| 真 SAR 适应 | Ours+AMS：8 epochs、encoder frozen、仅使用 train/val Noisy |
| 正式评价范围 | 合成 SAR 只报告 PSNR/SSIM；AMS 前后只评价真实 SAR；不测复杂度与推理时延 |

NWPU 与 UCM 是光学遥感图像，在论文中必须写成 *optical remote-sensing images corrupted with simulated SAR-like speckle*，不可称真实 SAR。真 SAR 数据单独来自 Mendeley `fs455tz88y.1`。

---

## 2. 双 GPU 的正确使用方式

本项目正式配置不使用 DDP，也不把两张 24 GiB 显存相加。每个训练进程只看见一张卡：

```text
physical GPU 0 -> one independent batch-1 run at a time
physical GPU 1 -> one independent batch-1 run at a time
```

正式并行队列由 `03_train_core_dualgpu.sh` 启动：

| GPU | 默认串行队列 |
|---:|---|
| 0 | Ours Full → Ours Log-only；启用 SAR-CAM 时再跑 SAR-CAM |
| 1 | TransSARV2 → Ours Intensity-only → Ours w/o-all-FDR |

脚本对子进程设置 `CUDA_VISIBLE_DEVICES=0` 或 `1`。因此每个子进程内部看到的设备都是逻辑 `cuda:0`，这是正常现象。不要把 `SAR_DEVICE` 写成 `cuda:1`，也不要再用 `torchrun`、`DataParallel` 或手工把 batch size 改为 2，否则会改变冻结协议。

16 vCPU 在两个训练进程间按每个进程 4 个 DataLoader workers、4 个 OMP/MKL threads 分配，给系统、校验和 I/O 留出余量。若远端实测 I/O 拥塞，可统一把 `SAR_WORKERS` 降到 2；这不会改变冻结样本流。

---

## 3. CUDA 版本的解释与环境验收

必须分别记录三件事：

- `nvidia-smi` 的 Driver Version：预期 `560.35.03`；
- `nvidia-smi` 顶部的 CUDA Version：这是驱动宣告的最高兼容能力，页面标注 12.6 不代表容器内装了 CUDA Toolkit 12.6；
- `torch.version.cuda`：PyTorch wheel/build 使用的 CUDA，本文环境预期为 `12.4`。

NVIDIA 官方说明 `nvidia-smi` 报告的是驱动所支持的最高 CUDA 版本；较新的驱动可运行以较旧 Toolkit 构建的程序。PyTorch 官方也确实发布了 Python 3.12 对应的 `torch==2.5.1`、`torchvision==0.20.1`、CUDA 12.4 构建。参见 [NVIDIA CUDA compatibility matrix](https://docs.nvidia.com/datacenter/tesla/drivers/cuda-toolkit-driver-and-architecture-matrix.html) 和 [PyTorch previous versions](https://pytorch.org/get-started/previous-versions/)。

先在服务器执行用户指定的原始核验：

```bash
nvidia-smi
lscpu
free -h
df -h
python --version

python - <<'PY'
import torch

print("PyTorch:", torch.__version__)
print("PyTorch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("Visible GPU count:", torch.cuda.device_count())

for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    print(
        f"GPU {i}: {props.name}, "
        f"VRAM: {props.total_memory / 1024**3:.2f} GiB"
    )
PY
```

注意 Python 属性必须是 `torch.__version__`，不是富文本误写的 `torch.**version**`。

代码上传且完成第 5.2 节的路径配置后，运行可归档的机器盘点：

```bash
source /REMOTE/CONFIG/icsps2026.env
cd "$SAR_PROJECT_ROOT"
bash scripts/icsps2026/00_server_inventory.sh \
  "$SAR_DATA_WORK_ROOT/server_inventory_configured"
```

它会保存 `nvidia-smi`、`lscpu`、`nproc`、cgroup CPU/内存限制、`free`、`df`、`lsblk` 和 PyTorch 可见 GPU 列表。容器里的 `lscpu` 可能显示宿主机全部逻辑核，而 `cpu.max` 才反映 16-vCPU 配额；二者都保留。CPU 型号若与网页的 Xeon Gold 6430/Platinum 8352V 描述不一致，论文环境说明采用实例内实测型号，并把“16 vCPU quota”与宿主拓扑分开写。

停止条件：PyTorch 未看到恰好两张 RTX 4090、任一卡显存低于约 23 GiB、Python/PyTorch/CUDA build 不符时，不要开始正式 100k run。

---

## 4. 存储规划与迁移清单

### 4.1 不要整体上传当前 45 GB `datasets/`

本机旧数据中约 41 GB 是本协议不使用的历史合成数据：

- `NWPU_RESISC45_SAR_global_L_v1`：约 25 GB；
- `NWPU_RESISC45_SAR_intensity_v1`：约 16 GB；
- `UCMerced_LandUse_SAR_mat`：旧协议约 1 GB；
- BSDS500 及旧合成集：本论文不使用。

远端只需要：

| 必需内容 | 本机约占用 | 远端角色 |
|---|---:|---|
| 原始 `NWPU-RESISC45` | 467 MB | 从中按 canonical manifest 读取 900/225 sources |
| 原始 UCMerced archive/images | 约 0.3 GB | 2,100-source 外部测试；建议服务器端校验下载 |
| canonical NWPU master manifest | 20 MB | 锁定 repaired 560/140 master，再确定 20/5 子集 |
| real SAR Noisy patches + grouped split | 约 2.2 GB（本机目录含无用 `.tmp`） | AMS 与真实测试 |
| GT16 | 不传输 | 用户已在 test 解锁前决定不使用；论文不报告任何 GT16-based quasi-reference metric |
| 新 fixed Gamma pairs | 约 4.6 GB | 900 NWPU validation + 8,400 UCM test |
| checkpoints/results | 取决于 seeds；核心单 seed 预留约 6–10 GB | best/last、日志、评测与图像 |

建议在正式准备前确认所选数据文件系统至少有 25 GiB 可用；若要同时保留 SAR-BM3D 输出、三 seeds 和大量真 SAR图像，建议留出 30 GiB 以上。30 GB 系统盘和 50 GB 数据盘的“标称容量”都不能替代 `df -hT` 的实际可用空间。

### 4.2 先确认数据盘挂载点

不要猜 `/data`、`/mnt/data` 或 `/workspace`。先运行：

```bash
lsblk -o NAME,SIZE,FSTYPE,MOUNTPOINTS,MODEL
df -hT
```

在输出中人工选定确实位于 50 GB SSD 上的实验根目录，后续把它写入 `SAR_DATA_WORK_ROOT`。本手册统一用 `/REMOTE/DATA` 表示这个**完整根目录本身**；它与环境文件中替换 `/SET/DATA_WORK_ROOT` 后的值必须完全相同，不要再多加一层 `sar_icsps26`。若平台没有挂载数据盘，先在平台控制面板完成挂载；不要让数据生成过程落到 30 GB 系统盘。

用真实挂载路径替换下面占位符后，先在服务器创建传输目标；否则后面以文件名为目标的 `rsync` 会因父目录不存在而失败：

```bash
# 只安装传输、校验和会话工具，不触碰 NVIDIA/CUDA/PyTorch。
# 无 sudo 时先用平台文件上传功能传代码，再请管理员补齐缺失命令。
sudo apt-get update
sudo apt-get install -y \
  git curl unzip tar gzip rsync jq tmux coreutils util-linux procps

mkdir -p /REMOTE/STAGING /REMOTE/PROJECT /REMOTE/CONFIG \
  /REMOTE/DATA/raw /REMOTE/DATA/seed_manifests
```

### 4.3 本机打包代码

在本机执行：

```bash
cd /Users/tim/sar_transformer-main
bash scripts/icsps2026/make_server_code_bundle.sh \
  /Users/tim/Downloads/sar_transformer_icsps26_code.tar.gz
```

该包刻意排除 `.git`、全部 datasets、checkpoints、results 和 tmp。随后用平台文件上传功能，或在取得 SSH 信息后用占位命令：

```bash
rsync -avP \
  /Users/tim/Downloads/sar_transformer_icsps26_code.tar.gz \
  /Users/tim/Downloads/sar_transformer_icsps26_code.tar.gz.sha256 \
  USER@HOST:/REMOTE/STAGING/
```

SSH 地址、端口和用户名尚未提供；不要原样执行 `USER@HOST`。如果平台使用非 22 端口，为 rsync 增加 `-e 'ssh -p PORT'`。

### 4.4 单独传输两个冻结 manifest

canonical master 位于被 `.gitignore` 忽略的 `datasets/`，干净 clone 或代码 tar 包都没有它。它必须单独传输并校验：

```bash
rsync -avP \
  /Users/tim/sar_transformer-main/datasets/NWPU_RESISC45_SAR_global_L_v2/dataset_manifest.json \
  USER@HOST:/REMOTE/DATA/seed_manifests/nwpu_global_L_v2_dataset_manifest.json

rsync -avP \
  /Users/tim/sar_transformer-main/datasets/real_sar_dataset/real_split_grouped_seed42.json \
  USER@HOST:/REMOTE/DATA/seed_manifests/real_split_grouped_seed42.json
```

远端必须得到：

```text
748b7010bf4f4c29eac5c814e32ba0f360e49ae7e063a20e03981a231c75e9bb  nwpu_global_L_v2_dataset_manifest.json
3d2161938861dddd66aed2950160b355bf067143564a35af51f6209dba659ec0  real_split_grouped_seed42.json
```

校验命令：

```bash
sha256sum \
  /REMOTE/DATA/seed_manifests/nwpu_global_L_v2_dataset_manifest.json \
  /REMOTE/DATA/seed_manifests/real_split_grouped_seed42.json
```

canonical master hash 不匹配时，正式数据准备会主动失败；不要重新生成一个“看起来也是 20/5”的替代 manifest。

### 4.5 传输原始 NWPU 和当前真 SAR Noisy

```bash
rsync -avP \
  /Users/tim/sar_transformer-main/datasets/NWPU-RESISC45/ \
  USER@HOST:/REMOTE/DATA/raw/NWPU-RESISC45/

rsync -avP \
  --exclude='/test_selected/***' \
  --include='*/' --include='*.mat' --exclude='*' \
  /Users/tim/sar_transformer-main/datasets/real_sar_dataset/ \
  USER@HOST:/REMOTE/DATA/raw/real_sar_dataset/
```

第二条命令只传冻结 manifest 所需的 `train/`、`val/`、`test/` 下 `.mat`：它会排除全部历史 `.tmp` 文件，也排除 `test_selected/` 中 120 个重复的人工预览副本。传输后至少核对：

```bash
find /REMOTE/DATA/raw/NWPU-RESISC45 -type f | wc -l
find /REMOTE/DATA/raw/real_sar_dataset \
  \( -path '*/train/*.mat' -o -path '*/val/*.mat' -o -path '*/test/*.mat' \) \
  -type f | wc -l
```

预期分别为 31,500 和 5,876。数量正确仍不等于内容正确，正式准备/QC 会继续做路径、hash、shape、finite 和 constant 检查。

### 4.6 UCM 与 no-GT16 决策

UCM 建议在服务器从 TorchGeo 固定镜像下载。该命令要在第 5.1–5.2 节的代码解包和环境文件配置完成后执行：

```bash
cd /REMOTE/PROJECT/sar_transformer-main
bash scripts/icsps2026/download_ucm_pinned.sh /REMOTE/DATA/raw
test -d "$UCM_ROOT"
test "$(find "$UCM_ROOT" -type f | wc -l)" -eq 2100
```

脚本锁定 TorchGeo 所列 archive 和 MD5 `5b7ec56793786b6dc8a908e8854ac0e4`。TorchGeo 文档确认其为 21 类、每类 100 张，并公布了同一下载地址与校验值：[TorchGeo UCMerced source](https://docs.torchgeo.org/en/v0.5.2/_modules/torchgeo/datasets/ucmerced.html)。

本论文已在任何真实 test 输出产生前冻结为 **不使用 GT16**。不要下载或传输 GT16；环境文件固定 `REAL_USE_GT16=0` 并清除 `REAL_GT16_ROOT`。不生成、不填写、不在论文表头预留任何 GT16-based quasi-reference metric。

---

## 5. 远端安装与环境变量

### 5.1 解包代码

```bash
mkdir -p /REMOTE/PROJECT
cd /REMOTE/STAGING
sha256sum -c sar_transformer_icsps26_code.tar.gz.sha256
tar -xzf /REMOTE/STAGING/sar_transformer_icsps26_code.tar.gz -C /REMOTE/PROJECT
cd /REMOTE/PROJECT/sar_transformer-main
```

### 5.2 配置路径

复制模板到仓库外，把 `/SET/PROJECT_ROOT` 替换为解包位置的父目录，把 `/SET/DATA_WORK_ROOT` 替换为第 4.2 节已核实的数据盘实验根目录（即手册中的 `/REMOTE/DATA`）：

```bash
cp scripts/icsps2026/env.server.example /REMOTE/CONFIG/icsps2026.env
${EDITOR:-vi} /REMOTE/CONFIG/icsps2026.env
source /REMOTE/CONFIG/icsps2026.env
```

关键变量应满足：

```bash
test -d "$SAR_PROJECT_ROOT"
test -d "$NWPU_ROOT"
test -d "$SAR_DATA_WORK_ROOT/raw"
test -f "$NWPU_MASTER_MANIFEST"
test -f "$REAL_SPLIT_MANIFEST"
df -hT "$SAR_DATA_WORK_ROOT"
```

此时 UCM 可能尚未下载，所以只检查其父目录；第 4.6 节下载后再检查 `$UCM_ROOT` 和 2,100 张图。

`PREP_ROOT`、`REAL_PROTOCOL_ROOT`、`RUN_ROOT` 必须在已确认的数据盘上。每次重新 `source` 模板时，`SMOKE_ROOT` 和 `PROFILE_ROOT` 会得到新的 UTC 时间戳。

现在创建正式记录副本，并重新执行第 3 节的 configured inventory：

```bash
mkdir -p "$RUN_ROOT/records"
cp -n records/icsps2026_templates/*.csv "$RUN_ROOT/records/"
cp -n docs/ICSPS2026_实验记录表.md "$RUN_ROOT/records/"
bash scripts/icsps2026/00_server_inventory.sh \
  "$SAR_DATA_WORK_ROOT/server_inventory_configured"
```

`cp -n` 是故意的：重新部署时不得覆盖已手工填写的正式记录。

### 5.3 安装新增依赖

不要执行旧 README 的 Python 3.6/PyTorch 1.7/mmcv-full 环境。新代码已经移除未使用的 mmcv 导入，服务器依赖文件也不会安装或替换 torch/torchvision：

先确保 Ubuntu 工具存在。第 4.2 节已执行过则这里不必重复；若当前用户有 `sudo` 权限，可执行：

```bash
sudo apt-get update
sudo apt-get install -y \
  git curl unzip tar gzip rsync jq tmux coreutils util-linux procps
```

若没有 `sudo`，用平台预装工具或联系管理员；不要因此自行替换 CUDA 镜像。`matlab` 只是 SAR-BM3D 所需的可选外部依赖，不由脚本安装。

```bash
cd "$SAR_PROJECT_ROOT"
bash scripts/icsps2026/00_install_server_deps.sh
bash scripts/icsps2026/00_preflight.sh
```

安装脚本会先要求 Python 3.12、PyTorch 2.5.1、`torch.version.cuda == 12.4`。由于已知服务器信息没有声明 torchvision，若它缺失或版本不是 0.20.1，脚本只会从 PyTorch 官方 `cu124` wheel 源安装匹配的 torchvision，不会替换 torch；随后再安装固定版本的实验依赖，并复核 torch/torchvision 未被间接更改。如果基础镜像的 torch 或 CUDA build 不匹配，优先重建/更换实例镜像；不要在一个已混杂的环境中静默替换 CUDA 栈。

---

## 6. 正式执行顺序

建议在 `tmux` 中执行所有长任务：

```bash
tmux new -s icsps26
source /REMOTE/CONFIG/icsps2026.env
cd "$SAR_PROJECT_ROOT"
set -o pipefail
```

### 6.1 P0-A：生成并完整验证合成数据

```bash
bash scripts/icsps2026/01_prepare_data.sh 2>&1 | tee "$SAR_DATA_WORK_ROOT/prepare_data.log"
```

正式 `--verify-only` 默认重新构建 canonical 900/225 选择与 100k schedule，并对全部 9,300 个 MAT 做内容级复核：文件 hash、数组 hash、L、seed、Gamma 重放和 clean source 映射。预期核心计数：

```text
NWPU train sources             900
NWPU validation sources        225
train schedule rows        100,000
NWPU fixed validation pairs    900
UCM fixed test pairs         8,400
L=1/2/4/8 schedule rows      25,000 each
D4 transform rows            12,500 each
```

生成中断时会留下隐藏的 `.<PREP_ROOT-name>.INCOMPLETE` staging 目录，正式目录不会被伪装成成功。先保存错误日志并检查原因，再把 staging 移到单独的诊断目录后重跑；不要直接覆盖一个已经成功发布的 `PREP_ROOT`。

`cross_dataset_phash_audit.csv` 只列候选的感知相似对，不会自动删除。在 test 解锁前必须查看候选数和每对图像，把人工结论与该 CSV 的 SHA-256 写入 `data_audit.csv`。如确认 NWPU–UCM 泄漏，必须升协议版本并统一重生，不能静默排除后继续。

### 6.2 P0-B：冻结真实 SAR QC

本论文固定执行 no-GT16 分支：

```bash
export REAL_USE_GT16=0
unset REAL_GT16_ROOT
bash scripts/icsps2026/05_prepare_real.sh 2>&1 | tee "$SAR_DATA_WORK_ROOT/prepare_real.log"
```

脚本通过 `--disable-gt16` 禁止自动发现同名目录；如果同时设置了 `REAL_GT16_ROOT` 会直接失败。若将来改变决定，必须建立新协议、新 QC 根并重跑全部真实域评测，不能把它追加到本次冻结结果。

运行后必须人工检查：

```bash
jq '.gt16_root, .counts, .numeric_mapping, .alignment_policy, .enl_roi_policy' \
  "$REAL_PROTOCOL_ROOT/real_qc_manifest.json"
```

并打开 `real_alignment_audit.csv` 与 `real_enl_roi_manifest.csv`。本分支应确认 `gt16_root=null`、所有 q-reference 有效计数为 0；ENL ROI 必须依据 Noisy 一次冻结，并人工确认不是明显边缘/目标区域。

### 6.3 P0-C：端到端 smoke

```bash
bash scripts/icsps2026/02_smoke.sh 2>&1 | tee "$SAR_DATA_WORK_ROOT/smoke.log"
```

该流程为五个核心模型各跑 2 updates，并只用 4 个 **NWPU validation** pairs 和 4 个 **real validation** patches 检查 forward、checkpoint、strict load、real QC/ROI loader、合成/真实 evaluator 和图像导出。它不会在第 6.5.1 节预注册之前暴露 UCM 或 real-test 的模型结果；输出永久标记为 `formal_run=false`，不得抄入论文。

若 smoke 因 GPU OOM 失败，不要直接缩小正式图像尺寸或混用两卡显存；先用 `nvidia-smi` 检查残留进程。若确有显存压力，保持 batch=1，关闭并发后单卡复测并记录峰值。

### 6.4 P0-D：1,000-update 计时试跑

```bash
bash scripts/icsps2026/02b_profile_1000_updates.sh \
  2>&1 | tee "$SAR_DATA_WORK_ROOT/profile_1000.log"
```

它在 GPU 0 跑 Full、GPU 1 跑 TransSARV2，均为非正式 profile。用实际时间估算：

```text
training_only_lower_bound_hours ≈ profile_wall_seconds × 100 / 3600
matrix_wall_hours ≈ max(sum(GPU0 queue), sum(GPU1 queue))
estimated_cost ≈ billed_instance_hours × platform_actual_hourly_price
```

profile 只在 step 0/末次各评 40 pairs，而正式训练有 21 次×900-pair validation，所以上式只是训练段的粗略下界，不能当整个 run 的报价。正式预算要额外加上一次 900-pair validation 的实测时间×21，再留 I/O 余量。不要直接用页面的 ￥1.98/小时或 ￥2.08/小时计算，直到平台确认哪个价格适用于该双卡实例以及是否为整机总价。

### 6.5 P1：双卡核心训练矩阵

先只跑 seed 42：

```bash
export CORE_SEEDS="42"
export INCLUDE_SAR_CAM=0
bash scripts/icsps2026/03_train_core_dualgpu.sh
```

脚本可断点恢复：已存在 `completion.json` 的 run 会跳过；存在 `checkpoint_last.pth` 的 run 会 strict resume。若目录存在但没有可恢复 checkpoint，脚本会停止而不是覆盖。

另开一个终端监控：

```bash
source /REMOTE/CONFIG/icsps2026.env
cd "$SAR_PROJECT_ROOT"
watch -n 2 nvidia-smi
tail -F "$RUN_ROOT"/logs/train_*.log
```

TensorBoard 使用已开放的 HTTP 6006 端口：

```bash
tensorboard --logdir "$RUN_ROOT/train" --host 0.0.0.0 --port 6006
```

平台外部访问地址尚未提供。TensorBoard 本身没有强认证；只有在平台端口访问受保护时才绑定外网。6008 留作备用，不需要为了本实验同时启动第二个服务。

每个正式 run 完成后检查：

```bash
find "$RUN_ROOT/train" -name completion.json -print
jq '{formal_run,completed_updates,target_updates,best,checkpoint_best_sha256}' \
  "$RUN_ROOT/train/ours_full/seed42/completion.json"
```

必须满足 `formal_run=true`、`completed_updates=target_updates=100000`。

### 6.5.1 P0-E：test 解锁前预注册门

在第一次运行 `04_evaluate_synthetic.sh` 或 `07_evaluate_real.sh` **之前**，完成下列动作：

1. 仅根据 NWPU `step_metrics.csv` 把 primary non-ours comparator 和选择规则写入 `$RUN_ROOT/records/validation_selection.csv`；被选中的唯一一行必须填 `selection_role=strongest_non_ours_baseline`，`method` 使用程序标识 `transsar_v2` 或 `sar_cam`，并填齐 `seed/global_step/val_macro_mse/checkpoint_sha256/formal_run`。`global_step` 和 `val_macro_mse` 必须逐字符复制 best 行的原始值，不得按论文显示精度提前四舍五入；
2. 在未看任何模型 test 输出时，把 UCM source/crop 和 real parent/crop 写入 `$RUN_ROOT/records/figure_selection.csv`；
3. 用下面的一次性 seal 命令做语义核验并原子写入解锁时间、两个 CSV 的 hash 与 `pretest_seal.json`。不要手工创建这三个文件。

```bash
python seal_icsps2026_pretest.py \
  --run-root "$RUN_ROOT" \
  --prep-root "$PREP_ROOT" \
  --real-protocol-root "$REAL_PROTOCOL_ROOT"
python verify_icsps2026_pretest_gate.py --run-root "$RUN_ROOT"
```

seal 会把 baseline 行与实际 `completion.json`、`step_metrics.csv`、`checkpoint_best.pth` 及其 SHA-256 对齐；把 UCM 行解析到冻结的 8,400-pair manifest 并检查 class/L/256×256 crop；把 RealSAR 行解析到冻结 QC 并检查 test split、sample-parent、有效 ROI 与 crop；同时绑定 prepared-data、real-QC 和两个选择 CSV 的 hash。它还会拒绝已经含 UCM/real-test/SAR-BM3D/论文图输出的 `RUN_ROOT`，且不会覆盖既有 seal。若需要改选择，必须在任何 test 输出产生前改用一个新的 `RUN_ROOT`，不能删除旧 seal 后原地重写。

每个 UCM/real 正式 runner 都会重新执行该语义校验，并断言本次传入的 manifest/QC 正是 seal 绑定版本；正式 aggregate 和后续汇总也保存并交叉核对 seal hash。UCM 至少预注册 $L=1$ 和 $L=4$ 各一条并填齐 source/class/crop；RealSAR 至少预注册 `homogeneous` 与 `structured` 各一条，并填齐唯一 `sample_id`、对应 `parent_id` 和 crop。`sample_id` 锁定 parent 的四个 child patches 中具体哪一张，避免评测后再挑图。把自动生成的 `test_unlock_utc.txt` 时间抄入人工记录表。

默认快速路线把 TransSARV2 预先定义为 primary comparator，后续才跑的 SAR-CAM 只是 secondary comparison，不得在看过 UCM 后改称“strongest”。若你希望 SAR-CAM 也参与 strongest-baseline 候选，必须在本节 test 解锁之前先完成其 smoke、100k 训练和 NWPU validation，再仅根据 validation 冻结二者中的对手。

### 6.6 P1：UCM 全量合成评测与 Lee-MMSE

```bash
export EVAL_SEEDS="42"
bash scripts/icsps2026/04_evaluate_synthetic_dualgpu.sh \
  2>&1 | tee "$RUN_ROOT/logs/evaluate_ucm_seed42.log"
```

双卡 wrapper 只执行一次 Noisy/Lee-MMSE，再把五个核心学习配置分给两个独立的 24 GiB 单卡进程；不做 DDP 或显存合并。中断后原命令可重跑，已通过 artifact verifier 的输出会跳过。若只允许使用一张卡，改跑 `04_evaluate_synthetic.sh`。

Lee 的一个全局 window 只在 NWPU validation 的候选 `{5,7,9,11}` 中选择，随后以已知 nominal L 作为 oracle-L 参数评价 UCM；因此表中必须标为 `oracle-L`，不能与 blind learned model 混称同一输入条件。该冻结选择由脚本一次完成并写入 `nwpu_window_tuning.csv`，不属于 `validation_selection.csv` 的 learned-baseline 预注册行。

所有正式 UCM 方法都应输出恰好 8,400 行 `per_image.csv`，包含 `pair_id/source_id/class_name/L/psnr/ssim`。

### 6.7 P1：官方 SAR-BM3D v1.0

SAR-BM3D 是闭源、仅限 nonprofit use 的 MATLAB 包，脚本不会静默接受许可证。先阅读官方许可，再在合规时下载：

```bash
bash scripts/icsps2026/fetch_external_baselines.sh \
  --external-root "$SAR_EXTERNAL_ROOT" \
  --with-sarbm3d \
  --accept-sarbm3d-nonprofit-license
```

设置解包目录和 archive：

```bash
export SARBM3D_ROOT="$SAR_EXTERNAL_ROOT/SARBM3D_v10_linux64"
export SARBM3D_ARCHIVE="$SAR_EXTERNAL_ROOT/SARBM3D_v10_linux64.tar.gz"
command -v matlab
```

先用独立目录做 4-pair 兼容性测试，确认旧 MEX/MATLAB 包能在 Ubuntu 22.04 上运行：

```bash
bash scripts/icsps2026/04b_sarbm3d_smoke.sh \
  2>&1 | tee "$SAR_DATA_WORK_ROOT/sarbm3d_smoke.log"
```

该输出带 `formal_run=false`，不得进入论文。兼容性通过后，正式全量再执行：

```bash
bash scripts/icsps2026/04b_sarbm3d.sh \
  2>&1 | tee "$RUN_ROOT/logs/evaluate_sarbm3d.log"
```

官方 README 明确 v1.0 发布于 2013-07-31、入口函数为 `SARBM3D_v10`、用途为 multiplicative speckle，并声明 nonprofit-only：[GRIP SAR-BM3D v1.0 README](https://www.grip.unina.it/download/prog/SAR-BM3D/version_1.0/README.txt)。它给出的旧硬件 256×256 参考时间约为每图 25–50 秒，8,400 pairs 可能成为整个流程的最长步骤；务必先实测，不要据旧机器时间直接估算服务器费用。

作者函数把输入/输出定义为 *square-root intensity*。因此 wrapper 固定执行：`noisy intensity [0,1] → sqrt → SARBM3D_v10(amplitude,L) → square → raw intensity → clip [0,1] only for common metrics`。原始 amplitude/intensity 极值与裁剪比例均写入证据链；不要把 intensity 直接送进函数。运行脚本只为 MATLAB 子进程临时加入作者包内 `lib_opencv210/glnxa64`，并用 `ldd` 检查旧 OpenCV 依赖；不会执行作者的 `install_opencv.m` 去修改系统库。若旧 MEX 与 Ubuntu 22.04/当前 MATLAB 不兼容，保留 smoke 日志并将此基线如实标为未完成，不能换 Python BM3D 冒名顶替。

### 6.8 P1：AMS 与真实 SAR 评价

AMS 只适配 Ours Full seed 42：

```bash
bash scripts/icsps2026/06_train_ams.sh \
  2>&1 | tee "$RUN_ROOT/logs/train_ams_seed42.log"
```

其 best checkpoint 只根据固定 real-validation masks 上的 masked loss 选择，不读取 UCM 或其他合成 test 结果。不得拿 last checkpoint 冒充 AMS best，也不得在 real test 上挑 epoch。

随后在同一 frozen test IDs/ROIs 上评价 Noisy、TransSARV2、Ours Log-only、Ours-base、Ours+AMS：

```bash
bash scripts/icsps2026/07_evaluate_real.sh \
  2>&1 | tee "$RUN_ROOT/logs/evaluate_real_seed42.log"
```

为适配 50 GB 标称数据盘，该 runner 保存完整 592-patch 指标 CSV，但不为每个方法重复落盘全部 592 张中间图；论文所需的预注册样例由第 7.1 节 exporter 从 checkpoint 精确重建。

本论文不使用 GT16，因此不报告任何 GT16-based quasi-reference metric。真实 SAR 最终指标与表格以实验方案和记录表为准；Noisy 的恒等指标只作输入锚点，不参与优劣排名。

真实 Lee/SAR-BM3D 不运行，因为当前没有可从数据元数据预先锁定的 effective L，也没有准参考用于独立验证该设定。二者只保留在已知 nominal $L$ 的 UCM 合成主表。

### 6.9 P2：SAR-CAM 与三 seeds

先获取并锁定官方 SAR-CAM commit：

```bash
bash scripts/icsps2026/fetch_external_baselines.sh \
  --external-root "$SAR_EXTERNAL_ROOT" \
  --with-sar-cam
export INCLUDE_SAR_CAM=1
```

runner 强制检查 commit `ea5ee3bed00ab22735a7c87518fe5388c2d6c49a`，并按官方 `channels=128` 构造模型。官方仓库称其为论文的 official PyTorch implementation：[SAR-CAM repository](https://github.com/JK-the-Ko/SAR-CAM)。

正式 100k 前先用新的非正式目录覆盖外部 checkout、forward、checkpoint 和 4-pair evaluator 链：

```bash
export SMOKE_ROOT="$SAR_DATA_WORK_ROOT/smoke/icsps26_sarcam_$(date -u +%Y%m%dT%H%M%SZ)"
bash scripts/icsps2026/02_smoke.sh \
  2>&1 | tee "$SAR_DATA_WORK_ROOT/sarcam_smoke.log"
```

因 `INCLUDE_SAR_CAM=1`，这次 smoke 会在五个 core 配置后额外跑 SAR-CAM；全部数字仍不得入论文。

先在单独的 GPU 0 上补齐 SAR-CAM seed 42；不要把它与下面 seeds 43/44 的自定义队列混在一起：

```bash
export CORE_SEEDS="42"
export CORE_METHOD_SPECS="sar_cam:"
CUDA_VISIBLE_DEVICES=0 SAR_DEVICE=cuda \
  bash scripts/icsps2026/03_train_core.sh
unset CORE_METHOD_SPECS

export EVAL_SEEDS="42"
export EVAL_METHOD_SPECS="sar_cam:"
CUDA_VISIBLE_DEVICES=0 SAR_DEVICE=cuda \
  bash scripts/icsps2026/04_evaluate_synthetic.sh
unset EVAL_METHOD_SPECS
```

这两条命令之前仍必须已完成第 6.5.1 的 test 解锁预注册。

核心证据完成后再补 Full 与 TransSARV2 的 seeds 43/44；不要在截止日前让次要三 seed 阻塞 seed 42 主表：

```bash
export CORE_SEEDS="43 44"
export GPU0_METHOD_SPECS="ours:full"
export GPU1_METHOD_SPECS="transsar_v2:"
bash scripts/icsps2026/03_train_core_dualgpu.sh

export EVAL_SEEDS="43 44"
export GPU0_EVAL_METHOD_SPECS="ours:full"
export GPU1_EVAL_METHOD_SPECS="transsar_v2:"
bash scripts/icsps2026/04_evaluate_synthetic_dualgpu.sh

unset GPU0_METHOD_SPECS GPU1_METHOD_SPECS \
  GPU0_EVAL_METHOD_SPECS GPU1_EVAL_METHOD_SPECS
```

如需为其他消融补 seeds，必须继续复用对应 `train_schedule_seed43.csv`/`seed44.csv`，不得临时重新抽 noise。

完成 Full/TransSARV2 的 42/43/44 后，用下列命令生成 seed-level mean±sample-SD，并在每个 frozen pair 先跨三 seed 取均值，再以 UCM source 为 cluster 做 paired bootstrap：

```bash
python summarize_icsps2026_multiseed.py \
  --member ours="$RUN_ROOT/ucm/ours_full/seed42/per_image.csv" \
  --member ours="$RUN_ROOT/ucm/ours_full/seed43/per_image.csv" \
  --member ours="$RUN_ROOT/ucm/ours_full/seed44/per_image.csv" \
  --member transsar="$RUN_ROOT/ucm/transsar_v2/seed42/per_image.csv" \
  --member transsar="$RUN_ROOT/ucm/transsar_v2/seed43/per_image.csv" \
  --member transsar="$RUN_ROOT/ucm/transsar_v2/seed44/per_image.csv" \
  --compare ours:transsar \
  --bootstrap-samples 10000 \
  --output-dir "$RUN_ROOT/summary/ucm_multiseed_42_43_44"
```

输出 `ucm_multiseed_summary.csv`、`ucm_multiseed_paired_bootstrap.csv` 和带全部输入 hash 的 `summary_provenance.json`。单 seed 主表不得伪写为 run-level mean±SD；只有这个脚本验证三个正式 seed 齐全后才填该列。

---

## 7. 汇总主表与 paired bootstrap

第 6.5.1 节中 primary non-ours comparator 必须已在不看 UCM test 的前提下写入运行目录的 `validation_selection.csv`。然后才可汇总；例如预注册 TransSARV2 为对手：

```bash
python summarize_icsps2026_synthetic.py \
  --input noisy="$RUN_ROOT/ucm/noisy/per_image.csv" \
  --input lee="$RUN_ROOT/ucm/lee_mmse/per_image.csv" \
  --input transsar="$RUN_ROOT/ucm/transsar_v2/seed42/per_image.csv" \
  --input intensity_only="$RUN_ROOT/ucm/ours_intensity_only/seed42/per_image.csv" \
  --input log_only="$RUN_ROOT/ucm/ours_log_only/seed42/per_image.csv" \
  --input wout_all_fdr="$RUN_ROOT/ucm/ours_wout_all_fdr/seed42/per_image.csv" \
  --input ours="$RUN_ROOT/ucm/ours_full/seed42/per_image.csv" \
  --compare ours:transsar \
  --compare ours:wout_all_fdr \
  --compare ours:log_only \
  --bootstrap-samples 10000 \
  --output-dir "$RUN_ROOT/summary/ucm_seed42"
```

如果 SAR-BM3D/SAR-CAM 已完成，在首次执行汇总前分别增加：

```bash
--input sarbm3d="$RUN_ROOT/ucm/sar_bm3d_v1/per_image.csv"
--input sar_cam="$RUN_ROOT/ucm/sar_cam/seed42/per_image.csv"
```

这两行需放在同一条 `python summarize_icsps2026_synthetic.py ...` 命令中，不是单独的 shell 命令。若 SAR-CAM 在 test 前已预注册为 primary comparator，把 `--compare ours:transsar` 替换为 `--compare ours:sar_cam`；同时把复制后的 `ucm_paired_comparisons.csv` 前三行 `baseline_key` 从 `transsar` 改为 `sar_cam`。若它是 test 后才补的 secondary baseline，只进主表，不伪称预注册 primary CI。输出：

- `ucm_main_summary.csv`：逐 L 与 class×L macro；
- `ucm_paired_cluster_bootstrap.csv`：以 UCM source 为 cluster、保留同源四个 L 的 paired 95% CI；
- `summary_provenance.json`：输入路径与预声明比较对象。

再对真实 SAR 做配对 parent-cluster bootstrap：

```bash
python summarize_icsps2026_real.py \
  --input transsar="$RUN_ROOT/real/transsar_v2_base/seed42/real_per_patch.csv" \
  --input log_only="$RUN_ROOT/real/ours_log_only_base/seed42/real_per_patch.csv" \
  --input ours_base="$RUN_ROOT/real/ours_full_base/seed42/real_per_patch.csv" \
  --input ours_ams="$RUN_ROOT/real/ours_full_ams/seed42/real_per_patch.csv" \
  --compare ours_ams:ours_base \
  --compare ours_base:transsar \
  --compare ours_base:log_only \
  --bootstrap-samples 10000 \
  --output-dir "$RUN_ROOT/summary/real_seed42"
```

`real_paired_parent_bootstrap.csv` 以 148 个 parent 为 cluster，保留每个 parent 的 4 个 child patches。`summary_provenance.json` 必须把所有 GT16-based quasi-reference metrics 标为 skipped；它们不进入模板或论文。

评价器的底层兼容字段不是待填实验结果；no-GT16 分支必须保持 `has_valid_gt16=false`，不得把任何准参考值抄入记录表或论文。

输出目录必须不存在，脚本不会覆盖旧汇总。若改变输入集合或预注册对手，使用新目录并在 deviation log 说明原因。

### 7.1 生成论文图片证据

所有图片只消费第 6.5.1 节 seal 中预注册的样例/crop，并保存原始数组、显示图、输入/checkpoint hash、provenance 与 completion；脚本没有“按 test 指标挑最好图片”的入口。确保 GPU 0 空闲后，生成 UCM 图 4：

```bash
CUDA_VISIBLE_DEVICES=0 python export_icsps2026_ucm_figures.py \
  --run-root "$RUN_ROOT" \
  --prep-root "$PREP_ROOT" \
  --method transsar_v2="$RUN_ROOT/train/transsar_v2/seed42/checkpoint_best.pth" \
  --method ours_wout_all_fdr="$RUN_ROOT/train/ours_wout_all_fdr/seed42/checkpoint_best.pth" \
  --method ours_full="$RUN_ROOT/train/ours_full/seed42/checkpoint_best.pth" \
  --device cuda:0 \
  --output-dir "$RUN_ROOT/paper_figures/ucm_pre_registered"
```

若 SAR-BM3D 已正式完成，在同一命令增加 `--sarbm3d-root "$RUN_ROOT/sarbm3d_raw_ucm"`；若 test 解锁前已锁定并训练 SAR-CAM，再同时增加对应 `--method sar_cam=...` 和 `--sar-cam-root "$SAR_CAM_ROOT"`。

生成图 5 的真实 SAR/ratio/AMS 预注册面板：

```bash
CUDA_VISIBLE_DEVICES=0 python export_icsps2026_real_figures.py \
  --run-root "$RUN_ROOT" \
  --dataset-root "$REAL_ROOT" \
  --real-protocol-root "$REAL_PROTOCOL_ROOT" \
  --method transsar_v2="$RUN_ROOT/train/transsar_v2/seed42/checkpoint_best.pth" \
  --method ours_log_only="$RUN_ROOT/train/ours_log_only/seed42/checkpoint_best.pth" \
  --method ours_full="$RUN_ROOT/train/ours_full/seed42/checkpoint_best.pth" \
  --ams ours_full_ams="$RUN_ROOT/ams/ours_full/seed42/checkpoint_best.pth" \
  --device cuda:0 \
  --output-dir "$RUN_ROOT/paper_figures/real_pre_registered"
```

真实图 exporter 是单卡顺序推理，不会调用第二张卡或合并显存。灰度固定映射到 `[0,1]`，ratio 显示固定为 `[0.5,1.5]`；ratio 图仅作诊断，不是 RGPI/M-index，也不能替代 592-patch 全量评价。输出目录必须是全新的；如未成功完成 AMS，省略 `--ams`，并在图注与 deviation log 如实说明。
若 seal 中的 strongest baseline 是 SAR-CAM，真实图命令也必须增加 `--method sar_cam="$RUN_ROOT/train/sar_cam/seed42/checkpoint_best.pth" --sar-cam-root "$SAR_CAM_ROOT"`。

---

## 8. 必须完成的实验与优先级

| 优先级 | 实验 | 完成判据 | 论文位置 |
|:---:|---|---|---|
| P0 | 机器/软件/磁盘 inventory | 两张 4090 分别约 24 GiB；环境与挂载已归档 | Reproducibility |
| P0 | frozen data prepare + verify | 900/225、100k、900/8400、全 MAT 复核通过 | Experimental setup |
| P0 | real Noisy QC/ROI/no-GT16 assertion | parent overlap 0；相同 valid IDs/ROIs；`gt16_root=null` | Real-data protocol |
| P0 | five-model smoke + 1k profile | 无 forward/checkpoint/evaluator 错误 | 不进论文 |
| P1 | Ours Full seed 42 | 100k + best/last hashes | 主表 |
| P1 | TransSARV2 seed 42 | 真正原始 baseline，100k | 主表 |
| P1 | Intensity-only seed 42 | 100k | 消融表 |
| P1 | Log-only seed 42 | 100k | 消融表 |
| P1 | Full w/o all FDR seed 42 | 100k | 消融表 |
| P1 | UCM Noisy + learned 全量 | 每方法 8,400 pairs | 合成主表/曲线 |
| P1 | Lee-MMSE | NWPU 选一次 window，UCM oracle-L 全量 | 合成主表 |
| P1 | SAR-BM3D v1.0 | 官方包、archive hash、8,400 pairs | 合成主表 |
| P1 | Ours+AMS | 8 epochs；只用固定 real-validation masked loss 选择 | 真实主表 |
| P1 | Real Noisy/TransSAR/Log-only/Base/AMS | 相同 test patches/parents/ROIs | 真实主表/ratio 图 |
| P2 | SAR-CAM retrained | pinned official commit、100k、全 UCM | 强化横向比较 |
| P2 | Full/TransSAR seeds 43/44 | 每个均 100k；seed-level mean±sample SD | 稳健性 |

截止时间紧时，先完整闭环全部 P0 和 seed-42 P1。不要为了多跑 seed 而删掉 Intensity-only 或 w/o-all-FDR；它们分别支撑表示链与频域模块的核心因果证据。

---

## 9. 实验记录如何填写

人工总表见 `docs/ICSPS2026_实验记录表.md`，机器可读空表位于 `records/icsps2026_templates/`。按以下来源填写：

| 文件 | 必须填写的时间 | 手工填写内容 | 唯一结果来源 |
|---|---|---|---|
| `data_audit.csv` | 数据准备后、test seal 前 | 实际 source/parent/pair/L 数、缺失/重复/overlap、saturation、人工 pHash/ROI 审核结论、manifest hash；`saturation_rate_L*` 填对应 L 的 `saturation_rate_mean`，min/max 写入 notes | `$PREP_ROOT/prepare_summary.json`（固定 pair 的 `fixed_pairs.*.synthesis_by_L`）、正式 seed-42 `completion.json.training_synthesis_by_L`（NWPU train）、`artifact_hashes.json`、`cross_dataset_phash_audit.csv`；`$REAL_PROTOCOL_ROOT/real_qc_manifest.json`/CSV |
| `validation_selection.csv` | 核心训练完成后、test seal 前 | **恰好一个** strongest non-ours baseline：method、seed、best step、原始 val macro MSE、checkpoint hash、selection role | 对应 `$RUN_ROOT/train/<method>/seed42/step_metrics.csv`、`completion.json`、`checkpoint_best.pth` SHA-256 |
| `figure_selection.csv` | **第一次 UCM/real test 前** | UCM 的 L=1/L=4 source/class/crop；RealSAR 的 homogeneous/structured sample/parent/crop；`selected_before_test_unlock=TRUE` | `$PREP_ROOT/ucm_test_manifest.csv`、`$REAL_PROTOCOL_ROOT/real_qc_manifest.csv` 和仅查看输入图像的人工选择；由 seal 做语义校验 |
| `run_registry.csv` | 每个正式监督训练 run 完成时 | run identity、状态、GPU、开始/结束时间、best step/MSE、wall time、配置/manifest/schedule/checkpoint/completion hashes | 各监督 run 的 `run_config.json`、`step_metrics.csv`、`completion.json`，以及代码包 `.sha256`；AMS 不把 epoch 冒充 global step |
| `ucm_main_table.csv` | UCM 全量评价和汇总后 | 每方法每个 L 与 macro 的 pairs、PSNR、SSIM | `$RUN_ROOT/summary/ucm_seed42/ucm_main_summary.csv`；各 `ucm/.../aggregate.json` |
| `ucm_paired_comparisons.csv` | UCM 汇总后 | Ours vs 预注册对手、w/o-all-FDR、Log-only 的 raw difference、advantage 和 source-cluster 95% CI | `$RUN_ROOT/summary/ucm_seed42/ucm_paired_cluster_bootstrap.csv` |
| `ablation_table.csv` | UCM 完成后 | 四个变体的结构开关、macro PSNR/SSIM | `ucm_main_summary.csv` 的四个 Ours 变体 macro 行 |
| `real_main_table.csv` | 全部 real evaluation 后 | 每方法的预注册真实 SAR 指标、逐指标观测数/parent 数及 CI | 各 `$RUN_ROOT/real/<method>/seed42/aggregate.json` 及相应汇总文件 |
| `real_paired_comparisons.csv` | real 汇总后 | 三组预声明比较的 advantage、parent-cluster 95% CI、paired patches/parents | `$RUN_ROOT/summary/real_seed42/real_paired_parent_bootstrap.csv`；no-GT16 时应恰好 9 行 |
| `deviation_log.csv` | 发生异常决定时立即填写 | 失败、排除、协议变化、是否需要重跑、失效 runs；全程无事件则保持 header-only | 原始日志、失败目录和实际人工决定；不能事后凭记忆补写 |

其中前三张不是“实验全部完成后再补”的表。`validation_selection.csv` 和 `figure_selection.csv` 会被一次性 `pretest_seal.json` 绑定；先运行 test 再填写将无法形成合规证据链。no-GT16 决策下不得自行增加任何 GT16-based quasi-reference metric。

`docs/ICSPS2026_实验记录表.md` 还需同步填写四类不适合硬塞进上述 CSV 的展示记录：第 0 节服务器/软件/hash；第 2 节 smoke 通过状态（明确不进论文）；第 6 节 AMS 的 epoch 0–8、fixed real-val masked loss 与 selected epoch（来源 `$RUN_ROOT/ams/ours_full/seed42/epoch_metrics.csv` 和 `completion.json`）；第 9 节最终图片输出路径和 provenance hash。

这些模板应已在第 5.2 节复制。若此时运行目录中仍没有记录副本，必须在任何 UCM/real test 前补做，不直接改模板原件：

```bash
mkdir -p "$RUN_ROOT/records"
cp -n records/icsps2026_templates/*.csv "$RUN_ROOT/records/"
cp -n docs/ICSPS2026_实验记录表.md "$RUN_ROOT/records/"
```

严禁填写：smoke/profile 数字、中期报告 compound-noise 数字、旧 mixed-L 数字、其他论文表格中的异协议数字、舰船检测结果。

---

## 10. 失败恢复与常见错误

| 现象 | 正确处理 |
|---|---|
| `torch.cuda.device_count()==1/0` | 检查平台 GPU 分配、容器 `--gpus`/可见性和残留环境变量；正式双队列前必须恢复 2 |
| `nvidia-smi` 显示 12.6、Torch 显示 12.4 | 正常且应分别记录；前者是驱动能力，后者是 PyTorch build |
| 单进程只显示一张 `cuda:0` | 双队列脚本通过 `CUDA_VISIBLE_DEVICES` 隔离后的正常结果 |
| OOM | 查残留进程；保持 batch=1 和 256²；逐个单卡 smoke，不把两卡显存相加 |
| 数据盘不足 | 停止；确认挂载；不要转移旧 41 GB 合成数据；必要时减少保存的可视化而非删正式原始证据 |
| canonical manifest hash 不匹配 | 重新传输原 20 MB 文件；禁止自制替代 split |
| prepared root 已存在 | 先运行 verify；不要覆盖。协议更改必须新目录/新版本 |
| `.INCOMPLETE` 存在 | 保存日志、查清失败、把 staging 移到诊断位置后重新生成 |
| 训练断线 | 重启同一脚本；它从 `checkpoint_last.pth` strict resume |
| Python 合成/真实/Lee 评估中断 | 保存日志，将该单个方法的 partial output 目录移到诊断位置后重跑；不得手工补 `aggregate.json` |
| checkpoint identity/hash 不符 | 停止并核对 method/variant/schedule；不要 `strict=False` |
| AMS 无 eligible checkpoint | 如实记为失败；保留 base，不能用 test 选 epoch |
| 发现或误设 GT16 | 停止并保持 `REAL_USE_GT16=0`、`REAL_GT16_ROOT` unset；本论文已冻结不使用 GT16 |
| SAR-BM3D MEX 不兼容 | 保留日志与 archive hash；不要换第三方 Python BM3D 冒充官方 SAR-BM3D |

正式训练期间每隔一段时间保存：

```bash
nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw \
  --format=csv
df -hT "$SAR_DATA_WORK_ROOT"
```

不要运行会删除整个 workspace、数据根或实验根的递归清理命令。所有重跑使用新目录；确需清理时先用 `du -sh` 和绝对路径确认目标，并优先移动到可恢复的隔离目录。

---

## 11. 论文证据边界与权威入口

- ICSPS Special Session 5 明确包含 SAR/PolSAR signal and image processing；full paper 不少于 5 页双栏，超过 6 页加收版面费，当前官网截止日期为 2026-09-10，且采用双盲评审：[Special Session 5](https://www.icsps.org/special5.html)、[Submission Instruction](https://www.icsps.org/sub.html)、[Peer Review](https://www.icsps.org/peer_review.html)。
- TransSAR 主干必须归属公开的 TransSARV2 实现；本项目 runner 调用真正的 `TransSARV2` baseline，而不是关闭本文模块后的近似替代：[official TransSAR repository](https://github.com/malshaV/sar_transformer)。
- SAR-CAM 只使用作者 official repository 的 pinned source，并在本协议下重训，不抄其论文异协议数字：[official SAR-CAM repository](https://github.com/JK-the-Ko/SAR-CAM)。
- 真 SAR 版本锁定 Mendeley V1，页面明确列出 Noisy、GT1、GT16，并采用 CC BY 4.0；本文只传输和评价 Noisy，GT1/GT16 不下载、不读取：[Mendeley dataset](https://data.mendeley.com/datasets/fs455tz88y/1)。
- UCM 下载、类别数量和 checksum 使用 TorchGeo 可追溯镜像；原始发布页仍可作为数据来源说明：[UC Merced dataset page](https://vision.ucmerced.edu/datasets/)。

网页日期和会议要求可能变化。投稿当天再次检查 `sub.html` 与 EasyChair，并保存带 UTC/北京时间的页面截图。
