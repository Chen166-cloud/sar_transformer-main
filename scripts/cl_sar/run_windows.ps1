[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$InputPath,
    [Parameter(Mandatory=$true)][ValidateSet('intensity','amplitude')][string]$InputDomain,
    [string]$MatField = 'noisy',
    [ValidateSet('auto','cpu','cuda')][string]$Device = 'auto',
    [double]$Scale = 1.0,
    [string]$OutputPath
)
$ErrorActionPreference = 'Stop'
$projectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
$python = Join-Path $projectRoot '.venv-cl-sar/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $python)) { throw 'Run scripts/cl_sar/setup_windows.ps1 first.' }
$scaleText = $Scale.ToString('R', [Globalization.CultureInfo]::InvariantCulture)
$arguments = @((Join-Path $PSScriptRoot 'run_local.py'), '--input', $InputPath,
    '--input-domain', $InputDomain, '--mat-field', $MatField, '--device', $Device, '--scale', $scaleText)
if ($OutputPath) { $arguments += @('--output', $OutputPath) }
& $python @arguments
if ($LASTEXITCODE -ne 0) { throw "CL-SAR failed with exit code $LASTEXITCODE." }
