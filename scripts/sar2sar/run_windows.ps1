[CmdletBinding()]
param(
    [string]$InputPath = '',
    [ValidateSet('amplitude', 'intensity')][string]$InputDomain = 'amplitude',
    [string]$MatField = 'noisy',
    [int]$CropSize = 0,
    [int]$Stride = 64,
    [int]$Threads = 4,
    [string]$OutputDir = ''
)
$ErrorActionPreference = 'Stop'
$projectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
$python = Join-Path $projectRoot '.venv-sar2sar/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $python)) { throw 'Run scripts/sar2sar/setup_windows.ps1 first.' }
$arguments = @((Join-Path $PSScriptRoot 'run_local.py'), '--input-domain', $InputDomain,
    '--mat-field', $MatField, '--stride', $Stride, '--threads', $Threads)
if ($PSBoundParameters.ContainsKey('CropSize')) { $arguments += @('--crop-size', $CropSize) }
if ($InputPath) { $arguments += @('--input', (Resolve-Path -LiteralPath $InputPath).Path) }
if ($OutputDir) { $arguments += @('--output', [IO.Path]::GetFullPath($OutputDir)) }
& $python -X utf8 @arguments
if ($LASTEXITCODE -ne 0) { throw 'SAR2SAR inference failed.' }
