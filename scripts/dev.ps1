$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot

Write-Host "=== PPC10 WebUI 开发模式启动 ===" -ForegroundColor Cyan

$backendJob = Start-Job -ScriptBlock {
    param($root)
    Set-Location $root
    python ppc10.py --webui --debug
} -ArgumentList $ProjectRoot

$frontendJob = Start-Job -ScriptBlock {
    param($root)
    Set-Location "$root\webui"
    npx vite --host
} -ArgumentList $ProjectRoot

Write-Host ""
Write-Host "后端 (Flask):  http://127.0.0.1:5000" -ForegroundColor Green
Write-Host "前端 (Vite):   http://127.0.0.1:3000" -ForegroundColor Green
Write-Host ""
Write-Host "按 Ctrl+C 停止所有服务..." -ForegroundColor Yellow

try {
    while ($true) {
        Start-Sleep -Seconds 1
        $beState = Get-Job -Id $backendJob.Id | Select-Object -ExpandProperty State
        $feState = Get-Job -Id $frontendJob.Id | Select-Object -ExpandProperty State
        if ($beState -eq 'Failed' -or $feState -eq 'Failed') {
            Write-Host "服务异常退出" -ForegroundColor Red
            break
        }
        if ($beState -eq 'Completed' -and $feState -eq 'Completed') {
            break
        }
    }
}
finally {
    Stop-Job -Job $backendJob -ErrorActionSilentlyContinue
    Stop-Job -Job $frontendJob -ErrorActionSilentlyContinue
    Remove-Job -Job $backendJob -Force -ErrorActionSilentlyContinue
    Remove-Job -Job $frontendJob -Force -ErrorActionSilentlyContinue
    Write-Host "所有服务已停止" -ForegroundColor Yellow
}
