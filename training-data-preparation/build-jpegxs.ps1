$ErrorActionPreference = 'Stop'
$projectRoot = $PSScriptRoot
$repositoryRoot = Split-Path -Parent $projectRoot
$portableBin = Join-Path $repositoryRoot '.cache\rust-toolchain\bin'
$previousPath = $env:PATH
$previousCargoHome = $env:CARGO_HOME
try {
    if (Test-Path -LiteralPath (Join-Path $portableBin 'cargo.exe')) {
        $env:PATH = $portableBin + ';' + $env:PATH
        $env:CARGO_HOME = Join-Path $repositoryRoot '.cache\cargo'
    }
    if (-not (Get-Command cargo -ErrorAction SilentlyContinue)) {
        throw 'Rust 1.95+ is required. Install Rust, then rerun this script.'
    }
    $manifest = Join-Path $projectRoot 'tools\jpegxs-bridge\Cargo.toml'
    & cargo test --locked --manifest-path $manifest
    if ($LASTEXITCODE -ne 0) { throw 'Rust bridge tests failed.' }
    & cargo build --locked --release --manifest-path $manifest
    if ($LASTEXITCODE -ne 0) { throw 'Rust bridge build failed.' }
    $destination = Join-Path $projectRoot 'bin'
    New-Item -ItemType Directory -Path $destination -Force | Out-Null
    Copy-Item -LiteralPath (Join-Path $projectRoot 'tools\jpegxs-bridge\target\release\brainhole-jpegxs.exe') -Destination (Join-Path $destination 'brainhole-jpegxs.exe') -Force
    Copy-Item -LiteralPath (Join-Path $projectRoot 'tools\jpegxs-bridge\THIRD_PARTY_NOTICES.md') -Destination (Join-Path $destination 'brainhole-jpegxs-NOTICES.md') -Force
    Write-Output 'Built bin/brainhole-jpegxs.exe. Use encoder.oxideav.json from the project folder.'
} finally {
    $env:PATH = $previousPath
    $env:CARGO_HOME = $previousCargoHome
}
