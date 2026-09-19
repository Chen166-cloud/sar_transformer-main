# MuLoG-DRUNet 单通道本地推理

对应 Cristiano Ulondu Mendes、Loïc Denis、Charles Deledalle、Florence Tupin 的 *Robustness to Spatially Correlated Speckle in Plug-and-Play PolSAR Despeckling*，IEEE TGRS 2024，[DOI](https://doi.org/10.1109/TGRS.2024.3432180)。入口使用作者公开的相关噪声 `generic_model.pth`、原始 DRUNet 结构及 MuLoG/ADMM，不做训练。

## 当前状态（2026-09-18）

本地入口、独立运行环境和官方源码校验已完成；项目统一的 **18 项离线输入/CLI/域转换测试通过**。官方 `generic_model.pth` 已从固定官方 HTTPS blob 的首个 ZIP 成员提取，并核对成员路径、压缩流结束标志、精确长度、CRC32 与固化的 SHA-256。官方 MuLoG/ADMM 与 DRUNet 严格权重加载的 CPU 工程核验已通过。

当前安装进度以 `output/mulog_drunet_local/pending_install_status.json` 为准，日志为同目录 `pending_install.log`。已启动的旧下载由 `scripts/mulog_drunet/finish_pending_download.ps1` 接管：复用已有字节，在首成员齐备后只终止 PID、创建时间及 fetch 命令行均匹配的本次下载进程，随后离线提取，并执行 CPU 核验与已知 L=1 样本的 10 轮工程检查。无需重下已有前缀，不继续获取其余三套网络。

## 来源和安装

- [作者仓库](https://gitlab.telecom-paris.fr/ring/mulog-drunet)，固定提交 `f468573f5bd4d7d30065380b8580269bcefbec19`。
- 原样源码和模型保留在 `external/MuLoG-DRUNet/`，不纳入主项目 Git。仓库未提供明确 LICENSE，不能据公开下载推断商业再分发许可。
- `py_functions.zip` 为 26,954 字节，SHA-256 `74204053abd6c28b3b56bbf1205e5a60f8463f081499e9b5d70679d334534594`。
- **只获取最终使用的 `models/generic_model.pth`，不要求完整 `models.zip`。** 固定官方 HTTPS blob 端点的 URL 标识为 `eff7e4c921bdd05caf875b126e7a8432c6c4d3e8`。服务器不支持 Range，因此请求开始后仅顺序读取首个 ZIP 成员所需的 **121,302,437 字节**（约 115.68 MiB），立即关闭响应；不是完整 484,888,919 字节包。
- 已核对首成员 ZIP header：路径 `models/generic_model.pth`，DEFLATE 压缩长度 121,302,383 字节，原始长度 **130,581,503 字节**，CRC32 **1667635814**。提取时验证 header、DEFLATE 结束标记、精确长度及 CRC32，并记录实际 SHA256。
- 源码 zip 有固定 SHA-256 和完整 Git blob 校验。模型 SHA-256 已固化为 `20bc285c8710214003f72bdf5e48004c1accb63547d4a304c89714c385dff124`；运行入口会同时核对精确长度、CRC32、来源记录和该哈希。CRC32 用于验证 ZIP 成员解压完整性，SHA-256 用于后续本地一致性复核。
- 不宣称验证了完整模型包的 Git blob 或 SHA256。`generic_model.zip-prefix` / 现有 `models.zip.download` 只是压缩前缀缓存，**不是完整 ZIP**；推理只需要源码、`generic_model.pth` 和来源记录，不依赖完整模型包。
- `.venv-mulog-drunet` 通过 `--system-site-packages` 复用 `D:/develop/Anaconda/envs/pytorch_gpu/python.exe` 的 PyTorch/CUDA，仅在独立 venv 安装 SciPy 1.17.0、ypstruct 0.0.2。没有修改基础环境。

```powershell
Set-Location 'D:\research\sar_transformer-main'
& .\scripts\mulog_drunet\setup_windows.ps1
```

下载器通过操作系统排他锁阻止多个 setup 同时写入同一个临时文件，并显示下载进度。异常退出时锁自动释放，临时文件保留。发现尚未齐备的已有前缀时不会隐式清空重下，应先确认旧下载是否仍在进行；该端点不支持 HTTP Range 续传。默认新下载只读取首成员所需字节。

如果已有下载文件达到 121,302,437 字节，可完全离线提取（先由上述控制器停止旧整包进程，勿同时写同一文件）：

```powershell
& .\.venv-mulog-drunet\Scripts\python.exe .\scripts\mulog_drunet\fetch_official.py `
  --extract-prefix .\external\MuLoG-DRUNet\models.zip.download
```

该模式不访问网络，不修改原始前缀；只解压首成员并写来源记录。仅检查/安装小型官方源码可用 `--source-only`。

作者 requirements 包含 Python 3.8、旧 SciPy、sparsesvd、BM3D 等多通道/其他 denoiser 依赖。本入口仅支持二维 D=1 DRUNet，官方 D=1 PCA 分支不调用稀疏 SVD，故未安装 Windows 编译困难且不参与该路径的 sparsesvd/BM3D；若误调用这些分支会立即报错。

## 显式指定域和 looks

`--input-domain intensity|amplitude` 和 `--looks` **均无默认值且必填**。仅有 `.mat`、字段名 `noisy` 或数值落在 `[0,1]`，不能据此推断其物理域或视数。

以下是项目中明确以 L=1 合成的强度样本，适合检查工程接入：

```powershell
& .\scripts\mulog_drunet\run_windows.ps1 `
  -InputPath 'datasets/NWPU_RESISC45_SAR_global_L_v2/val/L1/airplane/airplane_00003.mat' `
  -InputDomain intensity -Looks 1 -MatField noisy `
  -Iterations 10 -Device cuda
```

支持二维 MAT/NPY 和单通道灰度图片，不转换 RGB，不支持 MAT v7.3/HDF5。所有数值原样读取，包括 uint8：**不会自动除以 255**。例如已知幅度数据的标尺是 255 时，显式使用 `-InputDomain amplitude -Scale 255`。

变换顺序为输入除以 `scale`，幅度输入再平方变为强度，调用 MuLoG；输出如需幅度则开方，再乘回 `scale`。`denoised.npy`、`result.mat` 恢复到与原始输入相同的域与单位，保留 float64，不裁剪。PNG 所有面板固定使用原始单位下 `[0,scale]`，超出范围仅在预览时裁剪。

官方 `mulog.py` 注释规定 SLC 的 `L=1`、MLC 的 `L=ENL`；本入口要求用户提供，不自行从输出 ENL 或图像内容调参。`--iterations` 为 ADMM 的 T，默认 10 对齐作者 notebook（函数本身默认 6）。Newton 内循环原样为 10，初始 beta 为 `1+2/L`，lambda 为 1。

`looks` 接受任意有限正数，保留作者实现本身的处理：D=1 的 log-channel 标准差为 `sqrt(polygamma(1,max(1,L)))`，而初始化和数据似然仍使用传入的 L。不把小于 1 的 L 静默改成 1，也不声称此类数值一定符合某幅数据的物理噪声。

## 兼容层和数值核验

官方源文件保持字节不变。兼容层的范围如下：

1. 直接实例化原始 DRUNet，`weights_only=True`、`strict=True` 加载完整参数，缓存一次；不改变网络运算。
2. 作者 `funcDRUnet` 在 CPU 分支没有创建 Tensor。本入口补充显式 CPU/CUDA Tensor 转换，保留作者 ceil-index 的 0.003/0.997 分位数归一化及逆变换。
3. `np.complex` 通过只在官方 `matrixfields` 内生效的 NumPy proxy 兼容，调用后还原，不给基础 NumPy 注入全局别名。
4. Newton 求解只将并行分片数固定为 4；各像素公式及迭代数保持不变。
5. DRUNet 要求边长为 8 的倍数。任意尺寸在每次 denoiser 调用时只向下/右 replicate padding，随后裁回原尺寸；MuLoG 输入统计仍在原始尺寸上计算。整幅尺寸已为 8 的倍数时不 padding。
6. 保留作者的零值替换（用输入最小正强度替换零）及固定种子 4242 的 1e-6 相对扰动；这些动作和每次内部分位数变换都写入 `run.json`。全零/非法数组、非正 looks/scale、非有限输出会报错。

```powershell
& .\.venv-mulog-drunet\Scripts\python.exe .\scripts\mulog_drunet\verify_local.py --device cpu
```

核验使用上述已知 L=1 合成图的 32×40 小块，比较独立复写的作者 wrapper 和整个官方 MuLoG 路径、缓存重复调用、原生线程分片与固定 4 分片；另检查 29×37 非整倍尺寸以及幅度/强度、scale 往返。CUDA 模式额外直接执行作者未修改的 `funcDRUnet`，并在整个 T=2 MuLoG 内保留作者逐次加载模型的行为，与缓存路径做数值比对。核验不等同复现论文完整测试表。

每次推理输出 `denoised.npy`、`result.mat`（noisy、denoised、residual）、`denoised.png`、`comparison.png`、`run.json`。记录包含源版本与校验和、显式域/looks、所有尺度变换、设备和耗时。输出目录非空时拒绝覆盖。
