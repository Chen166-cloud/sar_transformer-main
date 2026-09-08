[CmdletBinding()]
param(
    [string]$TrainRoot = '',
    [string]$ValidationRoot = '',
    [string]$OutputDir = '',
    [ValidateRange(1, 2147483647)][int]$Updates = 200,
    [ValidateRange(1, 1024)][int]$BatchSize = 2,
    [int]$CropSize = 64,
    [int]$TrainSamples = 512,
    [int]$ValidationSamples = 8,
    [int]$ValidationInterval = 50,
    [int]$Seed = 20260908,
    [string]$Device = 'auto',
    [string]$Resume = ''
)
$ErrorActionPreference = 'Stop'
$projectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
$python = Join-Path $projectRoot '.venv-sar-cam/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $python)) { throw 'Run scripts/sar_cam/setup_windows.ps1 first.' }
if (-not $TrainRoot) { $TrainRoot = Join-Path $projectRoot 'datasets/NWPU_RESISC45_SAR_global_L_v2/train' }
if (-not $ValidationRoot) { $ValidationRoot = Join-Path $projectRoot 'datasets/NWPU_RESISC45_SAR_global_L_v2/val' }
if (-not $OutputDir) {
    $OutputDir = Join-Path $projectRoot ('output/sar_cam_local/train_' + (Get-Date -Format 'yyyyMMdd_HHmmss_fff'))
}
$arguments = @((Join-Path $PSScriptRoot 'train_local.py'), '--train-root', $TrainRoot,
    '--val-root', $ValidationRoot, '--output', $OutputDir, '--updates', "$Updates",
    '--batch-size', "$BatchSize", '--crop-size', "$CropSize", '--train-samples', "$TrainSamples",
    '--val-samples', "$ValidationSamples", '--validation-interval', "$ValidationInterval",
    '--seed', "$Seed", '--device', $Device)
if ($Resume) { $arguments += @('--resume', $Resume) }
& $python @arguments
if ($LASTEXITCODE -ne 0) { throw "SAR-CAM training failed (exit $LASTEXITCODE)." }
Write-Host "Local training results: $OutputDir"
