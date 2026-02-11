param(
    [string]$Version = ""
)

$ErrorActionPreference = "Stop"

function Get-VersionFromGit {
    try {
        $tag = git describe --tags --abbrev=0 2>$null
        if ($LASTEXITCODE -ne 0) { return $null }
        return $tag.Trim()
    } catch {
        return $null
    }
}

if ([string]::IsNullOrWhiteSpace($Version)) {
    $tag = Get-VersionFromGit
    if ($tag) { $Version = $tag }
}

if ([string]::IsNullOrWhiteSpace($Version)) {
    $Version = "1.0.0"
}

if ($Version.StartsWith("v")) {
    $Version = $Version.Substring(1)
}

$root = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$distDir = Join-Path $root "dist"
$wxsOut = Join-Path $PSScriptRoot "Files.wxs"
$msiOut = Join-Path $distDir "AirfoilFitter-$Version.msi"

Push-Location $root
try {
    pyinstaller AirfoilFitter.spec --noconfirm --clean
    $configSource = Join-Path $root "airfoilfitter.config.json"
    $configDest = Join-Path $distDir "AirfoilFitter\\airfoilfitter.config.json"
    if (Test-Path $configSource) {
        Copy-Item -Path $configSource -Destination $configDest -Force
    } else {
        throw "Missing config template: $configSource"
    }
} finally {
    Pop-Location
}

Push-Location $PSScriptRoot
try {
    python .\generate_wxs_fragment.py --source-dir $distDir\AirfoilFitter --output $wxsOut
    wix build .\AirfoilFitter.wxs $wxsOut -ext WixToolset.UI.wixext -ext WixToolset.Util.wixext -d Version=$Version -o $msiOut
} finally {
    Pop-Location
}

Write-Host "Built MSI: $msiOut"
