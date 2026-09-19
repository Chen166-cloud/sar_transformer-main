# Continue this installation's already-running legacy stream without restarting it.
# Stop at the generic ZIP member, extract only that model, then verify inference.
[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][int]$DownloaderProcessId,
    [Parameter(Mandatory=$true)][string]$DownloaderCreatedAt,
    [ValidateSet('cpu','cuda')][string]$VerifyDevice = 'cpu'
)
$ErrorActionPreference = 'Stop'
$projectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
$assetRoot = Join-Path $projectRoot 'external/MuLoG-DRUNet'
$prefix = Join-Path $assetRoot 'models.zip.download'
$python = Join-Path $projectRoot '.venv-mulog-drunet/Scripts/python.exe'
$outputRoot = Join-Path $projectRoot 'output/mulog_drunet_local'
$statusPath = Join-Path $outputRoot 'pending_install_status.json'
$targetBytes = 121302437L
$expectedCreatedAt = [DateTimeOffset]::Parse($DownloaderCreatedAt).UtcDateTime
$downloadGuard = $null
$controllerGuard = $null
$transcriptStarted = $false
$stage = 'starting'

function Get-VerifiedDownloader {
    $candidate = Get-CimInstance Win32_Process -Filter ('ProcessId=' + $DownloaderProcessId)
    if (-not $candidate) { return $null }
    $created = ([DateTime]$candidate.CreationDate).ToUniversalTime()
    $command = ([string]$candidate.CommandLine).Replace('\', '/')
    if ($created -ne $expectedCreatedAt -or
        -not $command.Contains('scripts/mulog_drunet/fetch_official.py') -or
        $candidate.Name -notlike 'python*.exe') {
        throw 'Downloader identity changed; refusing to operate on an unrelated process.'
    }
    return $candidate
}

function Save-InstallStatus([string]$State, [hashtable]$Details) {
    $record = [ordered]@{
        state = $State
        updated_at = (Get-Date).ToUniversalTime().ToString('o')
        target = 'models/generic_model.pth only'
        required_prefix_bytes = $targetBytes
        downloaded_prefix_bytes = $(if (Test-Path -LiteralPath $prefix) {
            (Get-Item -LiteralPath $prefix).Length
        } else { 0 })
        downloader_pid = $DownloaderProcessId
        controller_pid = $PID
        details = $Details
    }
    $record | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $statusPath -Encoding utf8
}

try {
    if (-not (Test-Path -LiteralPath $python)) { throw 'MuLoG runtime is missing.' }
    $null = New-Item -ItemType Directory -Path $outputRoot -Force
    # This stream predates fetch_official.py's lock. Hold the same lock while it
    # runs, so a repeated setup cannot overwrite its accumulated prefix.
    $controllerGuard = [IO.File]::Open((Join-Path $assetRoot '.pending-controller.lock'),
        [IO.FileMode]::OpenOrCreate, [IO.FileAccess]::ReadWrite, [IO.FileShare]::None)
    $downloadGuard = [IO.File]::Open((Join-Path $assetRoot '.download.lock'),
        [IO.FileMode]::OpenOrCreate, [IO.FileAccess]::ReadWrite, [IO.FileShare]::None)
    $null = Start-Transcript -Path (Join-Path $outputRoot 'pending_install.log') -Append
    $transcriptStarted = $true
    $stage = 'downloading_generic_prefix'
    while ($true) {
        $received = if (Test-Path -LiteralPath $prefix) { (Get-Item -LiteralPath $prefix).Length } else { 0L }
        Save-InstallStatus $stage @{}
        if ($received -ge $targetBytes) { break }
        if (-not (Get-VerifiedDownloader)) {
            throw 'The original transfer ended before the generic member was complete; partial data retained.'
        }
        Write-Host ('Generic model: {0:N1}/{1:N1} MiB received; keeping the existing transfer.' -f ($received / 1MB), ($targetBytes / 1MB))
        Start-Sleep -Seconds 10
    }
    # Stop only the exact task-owned process and its children, after checking
    # its PID, creation time, executable name and command. Do not fetch the rest.
    if (Get-VerifiedDownloader) {
        & taskkill.exe /PID $DownloaderProcessId /T /F
        if ($LASTEXITCODE -ne 0 -and (Get-VerifiedDownloader)) {
            throw 'Could not stop the old whole-archive transfer safely.'
        }
    }
    $downloadGuard.Dispose()
    $downloadGuard = $null
    $stage = 'extracting_generic_model'
    Save-InstallStatus $stage @{}
    & $python (Join-Path $PSScriptRoot 'fetch_official.py') --extract-prefix $prefix
    if ($LASTEXITCODE -ne 0) { throw 'Generic ZIP member verification or extraction failed.' }
    $stage = 'verifying_model'
    Save-InstallStatus $stage @{ device = $VerifyDevice }
    $verificationPath = Join-Path $outputRoot ('verification_pending_' + (Get-Date -Format 'yyyyMMdd_HHmmss_fff') + '.json')
    & $python (Join-Path $PSScriptRoot 'verify_local.py') --device $VerifyDevice --output $verificationPath
    if ($LASTEXITCODE -ne 0) { throw 'Real-weight parity verification failed; see transcript.' }
    $stage = 'running_known_intensity_sample'
    Save-InstallStatus $stage @{ device = $VerifyDevice; verification = $verificationPath }
    $runOutput = Join-Path $outputRoot ('project_synthetic_intensity_' + (Get-Date -Format 'yyyyMMdd_HHmmss_fff'))
    & $python (Join-Path $PSScriptRoot 'run_local.py') `
        --input (Join-Path $projectRoot 'datasets/NWPU_RESISC45_SAR_global_L_v2/val/L1/airplane/airplane_00003.mat') `
        --input-domain intensity --looks 1 --iterations 10 --device $VerifyDevice --output $runOutput
    if ($LASTEXITCODE -ne 0) { throw 'Known-intensity model smoke test failed; see transcript.' }
    $checkpoint = Join-Path $assetRoot 'models/generic_model.pth'
    Save-InstallStatus 'completed' @{
        checkpoint = $checkpoint
        checkpoint_sha256 = (Get-FileHash -LiteralPath $checkpoint -Algorithm SHA256).Hash.ToLowerInvariant()
        verification = $verificationPath
        run = (Join-Path $runOutput 'run.json')
        device = $VerifyDevice
        scope = 'Official generic member; local inference checks, not a paper benchmark.'
    }
    Write-Host 'Generic model downloaded, extracted, and verified. No other model was extracted.'
} catch {
    if ($controllerGuard -and (Test-Path -LiteralPath $outputRoot)) {
        Save-InstallStatus 'failed' @{ stage = $stage; error = $_.Exception.Message }
    }
    throw
} finally {
    if ($downloadGuard) { $downloadGuard.Dispose() }
    if ($controllerGuard) { $controllerGuard.Dispose() }
    if ($transcriptStarted) { $null = Stop-Transcript }
}
