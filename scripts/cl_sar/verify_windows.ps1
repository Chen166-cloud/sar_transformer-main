[CmdletBinding()]
param([ValidateSet('auto','cpu','cuda')][string]$Device = 'auto')
$ErrorActionPreference = 'Stop'
$projectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
$python = Join-Path $projectRoot '.venv-cl-sar/Scripts/python.exe'
& $python (Join-Path $PSScriptRoot 'verify_local.py') --device $Device
if ($LASTEXITCODE -ne 0) { throw "CL-SAR verification failed with exit code $LASTEXITCODE." }
