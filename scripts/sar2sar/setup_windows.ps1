[CmdletBinding()]
param(
    [string]$PythonExecutable = 'python',
    [string]$IndexUrl = 'https://pypi.tuna.tsinghua.edu.cn/simple'
)
$ErrorActionPreference = 'Stop'
$projectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
$environmentRoot = Join-Path $projectRoot '.venv-sar2sar'
$environmentPython = Join-Path $environmentRoot 'Scripts/python.exe'
& $PythonExecutable -c "import sys; assert sys.version_info[:2] == (3, 12), 'SAR2SAR local runtime requires Python 3.12'"
if ($LASTEXITCODE -ne 0) { throw 'Use a Python 3.12 interpreter with -PythonExecutable.' }
if (-not (Test-Path -LiteralPath $environmentPython)) {
    & $PythonExecutable -m venv $environmentRoot
    if ($LASTEXITCODE -ne 0) { throw 'Could not create the isolated Python environment.' }
}
& $environmentPython -m pip install --disable-pip-version-check --index-url $IndexUrl --require-hashes -r (Join-Path $PSScriptRoot 'requirements-win-py312.lock')
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
& $environmentPython (Join-Path $PSScriptRoot 'download_official.py')
if ($LASTEXITCODE -ne 0) { throw 'Official SAR2SAR snapshot download or verification failed.' }
Write-Host "SAR2SAR is ready. Python: $environmentPython"
