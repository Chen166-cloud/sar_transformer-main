# Historical Ours inference on the Umbra ROI

These runs use the historical `legacy_amplitude_v0` e47 Base and e8 AMS
checkpoints. They are **not** the current `intensity_v1` model or its
encoder-frozen AMS checkpoint. In the audited pair all 360 encoder state
tensors differ; the old `train_selfsup.py` enabled all parameters for training.

The input is the native 1024 x 1024 float32 linear-intensity ROI from
`output/cl_sar_umbra_buenos_aires/figure3_clsar_final/noisy_intensity.npy`.
Its SHA-256 is
`11d3f703329aaf133eb8560a84270736246119808032b626eb28bce6f6014b1e`.

The fixed explicit scale is `S = 120347.515625`, the observed peak for this
ROI shared by the two historical models and this multimethod experiment.
This is a transfer adapter, not radiometric calibration or a documented
historical training scale. The canonical legacy conversion is
`A = sqrt(I / S)`; restored intensity is `S * max(A_hat, 0)^2`.
No affine offset, percentile stretch, tile-specific normalization, resize,
8-bit quantization, or input clipping is applied. Legacy log/tanh/fusion
operators remain unchanged, with no new inverse-log operator inserted.

Inference uses 36 tiles of 256 x 256, overlap 64, stride 192, reflect padding,
and positive cosine weights. Raw amplitudes are blended before nonnegative
projection and squaring. This limits context and Fourier support to each
tile; it is not equivalent to a direct 1024 x 1024 forward.

Run from the repository root in PowerShell:

```powershell
& 'D:\develop\Anaconda\envs\pytorch_gpu\python.exe' -m venv --system-site-packages .venv-ours-legacy
& '.venv-ours-legacy\Scripts\python.exe' -m pip install -r scripts\ours_legacy\requirements-local.txt
& '.venv-ours-legacy\Scripts\python.exe' -B scripts\ours_legacy\run_umbra_roi.py --input output\cl_sar_umbra_buenos_aires\figure3_clsar_final\noisy_intensity.npy --output-dir output\umbra_buenos_aires_multimethod_1024\runs\ours_base_historical --variant base --intensity-scale 120347.515625 --device cuda
& '.venv-ours-legacy\Scripts\python.exe' -B scripts\ours_legacy\run_umbra_roi.py --input output\cl_sar_umbra_buenos_aires\figure3_clsar_final\noisy_intensity.npy --output-dir output\umbra_buenos_aires_multimethod_1024\runs\ours_ams_historical --variant ams --intensity-scale 120347.515625 --device cuda
```

Existing output directories are never overwritten; use a new directory for
a rerun. Each run contains the input, normalized intensity/amplitude, raw
network amplitude, nonnegative amplitude, and restored intensity as float32
NPY files, plus `result.mat` and provenance/validation in `run.json`.

The completed runs loaded all 531 state tensors strictly and produced finite
float32 1024 x 1024 arrays. MAT and NPY outputs are exactly equal; the input
is unchanged; the inverse conversion is exactly reproducible from the saved
raw amplitude. The identity stitching maximum absolute error is 2.98e-8.

The outputs expose a substantial transfer failure, especially for AMS:

| Historical model | Negative raw amplitude pixels | Restored mean / input mean |
| --- | ---: | ---: |
| Base e47 | 4.1454% | 0.3086 |
| AMS e8 | 98.1878% | 0.2171 |

Negative amplitudes are preserved in `network_raw_amplitude.npy`, then
projected to zero for physical intensity reconstruction. They are not made
positive by squaring. The near-all-black AMS result must not be presented as
a successful despeckling result or relabeled as the current paper's model.
