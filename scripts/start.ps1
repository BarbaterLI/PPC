$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot

Write-Host "=== PPC9 WebUI 生产模式启动 ===" -ForegroundColor Cyan

$webuiDist = Join-Path $ProjectRoot "webui" "dist"
if (-not (Test-Path $webuiDist)) {
    Write-Host "前端未构建，正在构建..." -ForegroundColor Yellow
    Push-Location (Join-Path $ProjectRoot "webui")
    npm run build
    Pop-Location
}

$env:FLASK_STATIC_FOLDER = $webuiDist

Set-Location $ProjectRoot
python -m src_m.web.run
