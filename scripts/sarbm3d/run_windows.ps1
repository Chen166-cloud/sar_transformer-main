[CmdletBinding()]
param(
    [string]$InputPath = '',
    [ValidateScript({ $_ -ge 1 -and -not [double]::IsInfinity($_) })]
    [double]$Looks = 1,
    [ValidateSet('intensity', 'amplitude')]
    [string]$InputDomain = 'intensity',
    [string]$MatField = 'noisy',
    [ValidateSet('official', 'synthetic')]
    [string]$Demo = 'official',
    [string]$OutputDir = '',
    [string]$PackageRoot = '',
    [string]$MatlabExecutable = 'matlab'
)
$ErrorActionPreference = 'Stop'
$projectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
if (-not $PackageRoot) {
    $PackageRoot = Join-Path $projectRoot 'external/sarbm3d/SARBM3D_v10_win64'
}
$PackageRoot = (Resolve-Path -LiteralPath $PackageRoot).Path
foreach ($required in @('SARBM3D_v10.m', 'SARBM3D_step1.mexw64', 'SARBM3D_step2.mexw64', 'removezeros.mexw64', 'cv210.dll', 'cxcore210.dll')) {
    if (-not (Test-Path -LiteralPath (Join-Path $PackageRoot $required) -PathType Leaf)) {
        throw "Missing $required in $PackageRoot. Run scripts/sarbm3d/setup_windows.ps1 first."
    }
}
if ($InputPath) { $InputPath = (Resolve-Path -LiteralPath $InputPath).Path }
if (-not $OutputDir) {
    $OutputDir = Join-Path $projectRoot ('output/sarbm3d_local/run_' + (Get-Date -Format 'yyyyMMdd_HHmmss_fff'))
}
$OutputDir = [IO.Path]::GetFullPath($OutputDir)
$null = New-Item -ItemType Directory -Path $OutputDir -Force
$matlabCommand = Get-Command $MatlabExecutable -ErrorAction Stop
# Environment values avoid inserting user paths into executable MATLAB text.
$parameters = @{
    SARBM3D_LOCAL_WRAPPER = (Join-Path $projectRoot 'matlab')
    SARBM3D_LOCAL_PACKAGE = $PackageRoot
    SARBM3D_LOCAL_OUTPUT = $OutputDir
    SARBM3D_LOCAL_INPUT = $InputPath
    SARBM3D_LOCAL_LOOKS = $Looks.ToString('R', [Globalization.CultureInfo]::InvariantCulture)
    SARBM3D_LOCAL_DOMAIN = $InputDomain
    SARBM3D_LOCAL_FIELD = $MatField
    SARBM3D_LOCAL_DEMO = $Demo
}
$previous = @{}
$previousPath = $env:PATH
try {
    foreach ($name in $parameters.Keys) {
        $previous[$name] = [Environment]::GetEnvironmentVariable($name, 'Process')
        [Environment]::SetEnvironmentVariable($name, $parameters[$name], 'Process')
    }
    $env:PATH = $PackageRoot + ';' + $env:PATH
    $expression = "addpath(getenv('SARBM3D_LOCAL_WRAPPER')); run_sarbm3d_local(getenv('SARBM3D_LOCAL_PACKAGE'),getenv('SARBM3D_LOCAL_OUTPUT'),getenv('SARBM3D_LOCAL_INPUT'),str2double(getenv('SARBM3D_LOCAL_LOOKS')),getenv('SARBM3D_LOCAL_DOMAIN'),getenv('SARBM3D_LOCAL_FIELD'),getenv('SARBM3D_LOCAL_DEMO'));"
    Write-Host "Running official SAR-BM3D v1.0, L=$Looks; results: $OutputDir"
    $logPath = Join-Path $OutputDir ('matlab_' + (Get-Date -Format 'yyyyMMdd_HHmmss_fff') + '.log')
    & $matlabCommand.Source -wait -nosplash -logfile $logPath -batch $expression
    if ($LASTEXITCODE -ne 0) { throw "MATLAB failed (exit $LASTEXITCODE). See $logPath" }
    if (-not (Test-Path -LiteralPath (Join-Path $OutputDir 'summary.json'))) {
        throw "MATLAB did not produce summary.json. See $logPath"
    }
    Get-Content -LiteralPath (Join-Path $OutputDir 'summary.json')
} finally {
    $env:PATH = $previousPath
    foreach ($name in $previous.Keys) {
        [Environment]::SetEnvironmentVariable($name, $previous[$name], 'Process')
    }
}
