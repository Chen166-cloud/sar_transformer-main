[CmdletBinding()]
param([string]$BasePython = 'D:\develop\Anaconda\envs\pytorch_gpu\python.exe')
$ErrorActionPreference = 'Stop'
$projectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
$package = Join-Path $projectRoot 'external/CL-SAR'
$runtime = Join-Path $projectRoot '.venv-cl-sar'
$python = Join-Path $runtime 'Scripts/python.exe'
$commit = 'b12129d1d3448750b9098b239397587eeb359857'
$null = Get-Command $BasePython -ErrorAction Stop
if (-not (Test-Path -LiteralPath $package)) {
    & git clone 'https://github.com/YangtianFang2002/CL-SAR-Despeckling.git' $package
    if ($LASTEXITCODE -ne 0) { throw 'Official CL-SAR clone failed.' }
    & git -C $package checkout --detach $commit
    if ($LASTEXITCODE -ne 0) { throw 'Could not select the pinned commit.' }
}
$installedCommit = & git -C $package rev-parse HEAD
if ($LASTEXITCODE -ne 0 -or $installedCommit.Trim() -ne $commit) { throw 'Existing source is not at the pinned commit.' }
$origin = & git -C $package remote get-url origin
if ($LASTEXITCODE -ne 0 -or $origin.Trim().TrimEnd('/').Replace('.git','') -ne 'https://github.com/YangtianFang2002/CL-SAR-Despeckling') { throw 'Unexpected source repository.' }
$changes = & git -C $package status --porcelain --untracked-files=no
if ($changes) { throw 'Official source has tracked changes; retaining it for inspection.' }
$weight = Join-Path $package 'experiments/MDN1-default/models/net_g_latest.pth'
if ((Get-FileHash -Algorithm SHA256 -LiteralPath $weight).Hash.ToLowerInvariant() -ne '94765b1e6dfb7584842dbad4b3683a566f77c5b3d85595dbb8913f90fef8c80b') { throw 'Official weight checksum mismatch.' }
if (-not (Test-Path -LiteralPath $python)) {
    & $BasePython -m venv --system-site-packages $runtime
    if ($LASTEXITCODE -ne 0) { throw 'Could not create local runtime.' }
}
& $python -m pip install -r (Join-Path $PSScriptRoot 'requirements-local.txt')
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
& $python -c "import torch, scipy, numpy, PIL; print('PyTorch:',torch.__version__,'CUDA:',torch.version.cuda,'GPU:',torch.cuda.is_available()); print('SciPy:',scipy.__version__,'NumPy:',numpy.__version__,'Pillow:',PIL.__version__)"
if ($LASTEXITCODE -ne 0) { throw 'Runtime validation failed.' }
Write-Host "CL-SAR runtime ready: $python"
