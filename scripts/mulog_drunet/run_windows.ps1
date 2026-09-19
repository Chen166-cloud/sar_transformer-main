[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$InputPath,
    [Parameter(Mandatory=$true)][ValidateSet('intensity','amplitude')][string]$InputDomain,
    [Parameter(Mandatory=$true)][double]$Looks,
    [string]$MatField = 'noisy',
    [double]$Scale = 1.0,
    [int]$Iterations = 10,
    [ValidateSet('auto','cuda','cpu')][string]$Device = 'auto',
    [string]$OutputDir = ''
)
$ErrorActionPreference = 'Stop'
$projectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
$python = Join-Path $projectRoot '.venv-mulog-drunet/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $python)) { throw 'Run scripts/mulog_drunet/setup_windows.ps1 first.' }
if (-not $OutputDir) {
    $OutputDir = Join-Path $projectRoot ('output/mulog_drunet_local/run_' + (Get-Date -Format 'yyyyMMdd_HHmmss_fff'))
}
$arguments = @((Join-Path $PSScriptRoot 'run_local.py'), '--input', $InputPath,
    '--input-domain', $InputDomain, '--looks', $Looks.ToString([Globalization.CultureInfo]::InvariantCulture),
    '--mat-field', $MatField, '--scale', $Scale.ToString([Globalization.CultureInfo]::InvariantCulture),
    '--iterations', $Iterations, '--device', $Device, '--output', $OutputDir)
& $python @arguments
if ($LASTEXITCODE -ne 0) { throw "MuLoG-DRUNet inference failed (exit $LASTEXITCODE)." }
Write-Host "Results: $OutputDir"
