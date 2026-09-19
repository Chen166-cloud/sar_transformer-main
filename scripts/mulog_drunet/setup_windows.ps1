[CmdletBinding()]
param([string]$BasePython = 'D:\develop\Anaconda\envs\pytorch_gpu\python.exe')
$ErrorActionPreference = 'Stop'
$projectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
$runtime = Join-Path $projectRoot '.venv-mulog-drunet'
$python = Join-Path $runtime 'Scripts/python.exe'
if (-not (Test-Path -LiteralPath $python)) {
    & $BasePython -m venv --system-site-packages $runtime
    if ($LASTEXITCODE -ne 0) { throw 'Could not create local runtime.' }
}
& $python -m pip install -r (Join-Path $PSScriptRoot 'requirements-local.txt')
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
& $python (Join-Path $PSScriptRoot 'fetch_official.py')
if ($LASTEXITCODE -ne 0) { throw 'Official asset verification failed.' }
& $python -c "import torch, scipy, numpy, ypstruct; print('PyTorch',torch.__version__,'GPU',torch.cuda.is_available(),'SciPy',scipy.__version__,'NumPy',numpy.__version__)"
if ($LASTEXITCODE -ne 0) { throw 'Runtime validation failed.' }
Write-Host "MuLoG-DRUNet runtime ready: $python"
