[CmdletBinding()]
param(
    [string]$InputPath = '',
    [string]$Weights = '',
    [string]$CheckpointMetadata = '',
    [string]$MatField = 'noisy',
    [string]$CleanPath = '',
    [string]$CleanMatField = 'clean',
    [ValidateSet('none', 'clip01', 'percentile')]
    [string]$Normalization = 'none',
    [string]$OutputDir = '',
    [string]$Device = 'auto'
)
$ErrorActionPreference = 'Stop'
$projectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
$python = Join-Path $projectRoot '.venv-sar-cam/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $python)) { throw 'Run scripts/sar_cam/setup_windows.ps1 first.' }
if (-not $InputPath) {
    $InputPath = Join-Path $projectRoot 'datasets/NWPU_RESISC45_SAR_global_L_v2/val/L1/airplane/airplane_00003.mat'
    if (-not $CleanPath) { $CleanPath = $InputPath }
}
if (-not $Weights) { $Weights = Join-Path $projectRoot 'output/sar_cam_local/training_demo/checkpoint_best.pth' }
if (-not (Test-Path -LiteralPath $Weights -PathType Leaf)) {
    throw 'Trained SAR-CAM weights are required. Run train_windows.ps1 or supply -Weights and its provenance metadata.'
}
if (-not $OutputDir) {
    $OutputDir = Join-Path $projectRoot ('output/sar_cam_local/run_' + (Get-Date -Format 'yyyyMMdd_HHmmss_fff'))
}
$arguments = @((Join-Path $PSScriptRoot 'run_local.py'), '--checkpoint', $Weights,
    '--input', $InputPath, '--mat-field', $MatField, '--normalization', $Normalization,
    '--output', $OutputDir, '--device', $Device)
if ($CleanPath) { $arguments += @('--clean', $CleanPath, '--clean-mat-field', $CleanMatField) }
if ($CheckpointMetadata) { $arguments += @('--checkpoint-metadata', $CheckpointMetadata) }
& $python @arguments
if ($LASTEXITCODE -ne 0) { throw "SAR-CAM inference failed (exit $LASTEXITCODE)." }
Write-Host "Results: $OutputDir"
