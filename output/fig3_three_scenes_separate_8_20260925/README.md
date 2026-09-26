# 图 3 式独立小图

共 3 个场景、每场景 8 张无图号和方法文字的方形 PNG。文件名 `a`–`h` 依次对应 Noisy、SAR-BM3D、SAR2SAR、SDUDNet、Trans-SAR、CL-SAR、MERLIN、MuLoG-DRUNet。可在 Word 中按每行 4 张、共 2 行插入；每张图片设为 **2.106 × 2.106 cm**，然后在 Word 里另行编辑 `(a)`–`(h)` 与方法名。这与 `D:\Users\Chen\Desktop\V3修改答复.docx` 的图 3 图片对象尺寸和“图文分离”形式一致。

- `01_Hong_Kong_ground_v3/`：来自用户指定的香港等地面比例中央方图；每张 502×502 像素，原方法输入仍是 1024×1024 SICD ROI。
- `02_Nevada_from_report_v2/`：来自用户指定旧报告中的内华达 Tesla 图；每张 1024×1024 像素，未做地面等比例重投影。
- `03_Melbourne_from_report_v2/`：来自用户指定旧报告中的墨尔本图；每张 1024×1024 像素，未做地面等比例重投影。

导出过程**没有裁切、缩放或重新拉伸灰度**；PNG 的像素与对应无字源单图逐像素相同，仅将 DPI 元数据设为接近参考 Word 的默认插入尺寸。香港与其余两景来自不同版本的 ROI 和显示几何，不能当成三幅完全相同协议的地面等比例图。逐文件来源、像素尺寸和校验见 [`manifest.json`](manifest.json)。
