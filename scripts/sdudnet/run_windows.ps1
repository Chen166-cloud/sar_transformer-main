[CmdletBinding()]
param(
    [string]$InputPath = '',
    [ValidateSet('real', 'synthetic')][string]$Model = 'real',
    [string]$MatField = 'noisy',
    [string]$CleanPath = '',
    [string]$CleanMatField = 'clean',
    [double]$Scale = 1.0,
    [double]$DataRange = 1.0,
    [ValidateSet('auto', 'cuda', 'cpu')][string]$Device = 'auto',
    [string]$OutputDir = ''
)
$ErrorActionPreference = 'Stop'
$projectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
$python = Join-Path $projectRoot '.venv-sdudnet/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $python)) { throw 'Run scripts/sdudnet/setup_windows.ps1 first.' }
if (-not $InputPath) { $InputPath = Join-Path $projectRoot 'external/SDUDNet/my_datasets/01233.jpg' }
if (-not $OutputDir) {
    $OutputDir = Join-Path $projectRoot ('output/sdudnet_local/run_' + (Get-Date -Format 'yyyyMMdd_HHmmss_fff'))
}
$arguments = @((Join-Path $PSScriptRoot 'run_local.py'), '--input', $InputPath,
    '--model', $Model, '--mat-field', $MatField, '--output', $OutputDir, '--device', $Device,
    '--scale', $Scale.ToString([Globalization.CultureInfo]::InvariantCulture),
    '--data-range', $DataRange.ToString([Globalization.CultureInfo]::InvariantCulture))
if ($CleanPath) { $arguments += @('--clean', $CleanPath, '--clean-mat-field', $CleanMatField) }
& $python @arguments
if ($LASTEXITCODE -ne 0) { throw "SDUDNet inference failed (exit $LASTEXITCODE)." }
Write-Host "Results: $OutputDir"
