[CmdletBinding()]
param([string]$BasePython = 'D:\develop\Anaconda\envs\pytorch_gpu\python.exe')
$ErrorActionPreference = 'Stop'
$projectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
$package = Join-Path $projectRoot 'external/SAR-CAM'
$runtime = Join-Path $projectRoot '.venv-sar-cam'
$python = Join-Path $runtime 'Scripts/python.exe'
$commit = 'ea5ee3bed00ab22735a7c87518fe5388c2d6c49a'
$null = Get-Command $BasePython -ErrorAction Stop
if (-not (Test-Path -LiteralPath $package)) {
    & git clone 'https://github.com/JK-the-Ko/SAR-CAM.git' $package
    if ($LASTEXITCODE -ne 0) { throw 'Official SAR-CAM clone failed.' }
    & git -C $package checkout --detach $commit
    if ($LASTEXITCODE -ne 0) { throw 'Could not select the verified source commit.' }
}
$installedCommit = & git -C $package rev-parse HEAD
if ($LASTEXITCODE -ne 0 -or $installedCommit.Trim() -ne $commit) {
    throw "Existing source is not at the verified commit $commit."
}
$changes = & git -C $package status --porcelain --untracked-files=no
if ($changes) { throw 'Author source has local tracked changes; retaining it for inspection.' }
if (-not (Test-Path -LiteralPath $python)) {
    & $BasePython -m venv --system-site-packages $runtime
    if ($LASTEXITCODE -ne 0) { throw 'Could not create isolated adapter environment.' }
}
& $python -m pip install -r (Join-Path $PSScriptRoot 'requirements-local.txt')
if ($LASTEXITCODE -ne 0) { throw 'Adapter dependency installation failed.' }
& $python -c "import torch, scipy, numpy, PIL; print('PyTorch:',torch.__version__,'CUDA:',torch.version.cuda,'GPU:',torch.cuda.is_available()); print('SciPy:',scipy.__version__,'NumPy:',numpy.__version__,'Pillow:',PIL.__version__)"
if ($LASTEXITCODE -ne 0) { throw 'Runtime validation failed. BasePython must provide working PyTorch.' }
Write-Host "SAR-CAM runtime ready: $python"
Write-Host 'The author repository supplies source code; train locally or provide a compatible trained checkpoint.'
