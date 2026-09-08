[CmdletBinding()]
param([string]$BasePython = 'D:\develop\Anaconda\envs\pytorch_gpu\python.exe')
$ErrorActionPreference = 'Stop'
$projectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
$package = Join-Path $projectRoot 'external/SDUDNet'
$runtime = Join-Path $projectRoot '.venv-sdudnet'
$python = Join-Path $runtime 'Scripts/python.exe'
$commit = '0c799911e84ef79c5fbe6f58f44cb8ee033048a1'
$null = Get-Command $BasePython -ErrorAction Stop
if (-not (Test-Path -LiteralPath $package)) {
    & git clone 'https://github.com/BFY-official/SDUDNet.git' $package
    if ($LASTEXITCODE -ne 0) { throw 'Official SDUDNet clone failed.' }
    & git -C $package checkout --detach $commit
    if ($LASTEXITCODE -ne 0) { throw 'Could not select the verified source commit.' }
}
$installedCommit = & git -C $package rev-parse HEAD
if ($LASTEXITCODE -ne 0 -or $installedCommit.Trim() -ne $commit) {
    throw "Existing source is not at the verified commit $commit."
}
$changes = & git -C $package status --porcelain --untracked-files=no
if ($changes) { throw 'Official source has tracked changes; retaining it for inspection.' }
if (-not (Test-Path -LiteralPath $python)) {
    & $BasePython -m venv --system-site-packages $runtime
    if ($LASTEXITCODE -ne 0) { throw 'Could not create the local runtime.' }
}
& $python -m pip install -r (Join-Path $PSScriptRoot 'requirements-local.txt')
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
& $python -c "import torch, scipy, numpy, PIL; print('PyTorch:',torch.__version__,'CUDA:',torch.version.cuda,'GPU:',torch.cuda.is_available()); print('SciPy:',scipy.__version__,'NumPy:',numpy.__version__,'Pillow:',PIL.__version__)"
if ($LASTEXITCODE -ne 0) { throw 'Runtime validation failed. BasePython must provide working PyTorch.' }
Write-Host "SDUDNet runtime ready: $python"
