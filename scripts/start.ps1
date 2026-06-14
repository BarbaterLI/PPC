$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot

Write-Host "=== PPC10 WebUI 生产模式启动 ===" -ForegroundColor Cyan

$webuiDist = Join-Path $ProjectRoot "webui" "dist"
if (Test-Path $webuiDist) {
    $env:FLASK_STATIC_FOLDER = $webuiDist
}

Set-Location $ProjectRoot
python ppc10.py --webui
