param([string]$TectonicPath)

$ErrorActionPreference = 'Stop'
$paperDir = $PSScriptRoot
$projectDir = Split-Path (Split-Path $paperDir -Parent) -Parent
if (-not $TectonicPath) {
    $compiler = Get-Command tectonic -ErrorAction SilentlyContinue
    if ($compiler) { $TectonicPath = $compiler.Source }
    else { $TectonicPath = Join-Path $projectDir 'tmp/latex_runtime/tectonic.exe' }
}
if (-not (Test-Path -LiteralPath $TectonicPath -PathType Leaf)) {
    throw 'Tectonic was not found. Supply -TectonicPath with its executable path.'
}
$TectonicPath = (Resolve-Path -LiteralPath $TectonicPath).Path
$buildDir = Join-Path $projectDir 'tmp/pdfs/grsl_conversion/build'
New-Item -ItemType Directory -Path $buildDir -Force | Out-Null
Push-Location -LiteralPath $paperDir
try {
    foreach ($document in @('GRSL_paper', 'GRSL_method_diagrams')) {
        & $TectonicPath -X compile "$document.tex" --outdir $buildDir --keep-logs --keep-intermediates
        if ($LASTEXITCODE -ne 0) { throw "Tectonic compilation failed for $document ($LASTEXITCODE)." }
        Copy-Item -LiteralPath (Join-Path $buildDir "$document.pdf") -Destination (Join-Path $paperDir "$document.pdf") -Force
        Write-Output (Join-Path $paperDir "$document.pdf")
    }
} finally {
    Pop-Location
}
