[CmdletBinding()]
param([string]$Destination = '')
$ErrorActionPreference = 'Stop'
$projectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
if (-not $Destination) { $Destination = Join-Path $projectRoot 'external/sarbm3d' }
$Destination = [IO.Path]::GetFullPath($Destination)
$null = New-Item -ItemType Directory -Path $Destination -Force
$url = 'https://www.grip.unina.it/download/prog/SAR-BM3D/version_1.0/SARBM3D_v10_win64.zip'
$expectedSha256 = '6bad387f5b085cb01a0559064812de6e667bd0b0cc784b4611f03520594e7da6'
$archive = Join-Path $Destination 'SARBM3D_v10_win64.zip'
$package = Join-Path $Destination 'SARBM3D_v10_win64'
Write-Host 'Official GRIP-UNINA SAR-BM3D v1.0: nonprofit use only; retain LICENSE.txt and author attribution. Do not redistribute the archive or extracted software.'
if (-not (Test-Path -LiteralPath $archive)) {
    $partial = $archive + '.part'
    & curl.exe --fail --location --retry 3 --connect-timeout 30 --max-time 900 --continue-at - --output $partial $url
    if ($LASTEXITCODE -ne 0) { throw "Download failed; partial download retained at $partial" }
    if ((Get-FileHash -LiteralPath $partial -Algorithm SHA256).Hash.ToLowerInvariant() -ne $expectedSha256) {
        throw 'Downloaded archive does not match the verified 2026-09-08 SHA-256; retained for inspection.'
    }
    Move-Item -LiteralPath $partial -Destination $archive
}
if ((Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash.ToLowerInvariant() -ne $expectedSha256) {
    throw 'Existing archive does not match the verified SHA-256.'
}
Add-Type -AssemblyName System.IO.Compression.FileSystem
$zip = [IO.Compression.ZipFile]::OpenRead($archive)
try {
    foreach ($entry in $zip.Entries) {
        $target = [IO.Path]::GetFullPath((Join-Path $Destination $entry.FullName))
        if (-not $target.StartsWith($Destination.TrimEnd('\') + '\', [StringComparison]::OrdinalIgnoreCase)) {
            throw "Unsafe archive path: $($entry.FullName)"
        }
    }
    if (-not (Test-Path -LiteralPath $package)) {
        [IO.Compression.ZipFile]::ExtractToDirectory($archive, $Destination)
    }
    # Verify the extracted files too; never silently overwrite a modified installation.
    foreach ($entry in $zip.Entries) {
        if (-not $entry.Name) { continue }
        $target = Join-Path $Destination $entry.FullName
        if (-not (Test-Path -LiteralPath $target -PathType Leaf)) { throw "Missing installed file: $target" }
        $stream = $entry.Open()
        $hasher = [Security.Cryptography.SHA256]::Create()
        try { $entryHash = ([BitConverter]::ToString($hasher.ComputeHash($stream))).Replace('-', '').ToLowerInvariant() }
        finally { $stream.Dispose(); $hasher.Dispose() }
        if ((Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash.ToLowerInvariant() -ne $entryHash) {
            throw "Installed file differs from official archive: $target"
        }
    }
} finally { $zip.Dispose() }
$fileHashes = @{}
Get-ChildItem -LiteralPath $package -File | ForEach-Object {
    $fileHashes[$_.Name] = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
}
$provenance = [ordered]@{
    implementation = 'GRIP-UNINA SAR-BM3D v1.0 (Windows x64)'
    release_date = '2013-07-31'
    source_url = $url
    readme_url = 'https://www.grip.unina.it/download/prog/SAR-BM3D/version_1.0/README.txt'
    paper_doi = '10.1109/TGRS.2011.2161586'
    archive_sha256 = $expectedSha256
    verified_at = (Get-Date).ToString('o')
    package_root = $package
    files_sha256 = $fileHashes
}
$provenance | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $Destination 'provenance.json') -Encoding UTF8
Write-Host "Verified installation: $package"
