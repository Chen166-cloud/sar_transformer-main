param([string]$TectonicPath)

$ErrorActionPreference = 'Stop'
$paperDir = $PSScriptRoot
$projectDir = Split-Path (Split-Path $paperDir -Parent) -Parent
if (-not $TectonicPath) {
    $compiler = Get-Command tectonic -ErrorAction SilentlyContinue
    if ($compiler) {
        $TectonicPath = $compiler.Source
    } else {
        $TectonicPath = Join-Path $projectDir 'tmp/latex_runtime/tectonic.exe'
    }
}
if (-not (Test-Path -LiteralPath $TectonicPath -PathType Leaf)) {
    throw 'Tectonic was not found. Supply -TectonicPath with its executable path.'
}
$TectonicPath = (Resolve-Path -LiteralPath $TectonicPath).Path
$buildDir = Join-Path $projectDir 'tmp/pdfs/paper_fig123/build'
New-Item -ItemType Directory -Path $buildDir -Force | Out-Null
Push-Location -LiteralPath $paperDir
try {
    & $TectonicPath -X compile 'ICSPS2026_paper.tex' --outdir $buildDir --keep-logs --keep-intermediates
    if ($LASTEXITCODE -ne 0) { throw "Tectonic compilation failed ($LASTEXITCODE)." }
    Copy-Item -LiteralPath (Join-Path $buildDir 'ICSPS2026_paper.pdf') -Destination (Join-Path $paperDir 'ICSPS2026_paper.pdf') -Force
    Write-Output (Join-Path $paperDir 'ICSPS2026_paper.pdf')
} finally {
    Pop-Location
}
