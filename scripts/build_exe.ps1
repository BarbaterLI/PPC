<#
.SYNOPSIS
    用 Nuitka 打包 PPC（standalone）→ compiled_output\ppc.dist\ppc.exe

.DESCRIPTION
    本项目有 3 类「静态分析看不见」的依赖，裸命令漏一个就是「编译成功、运行报错」：

    1. 包内数据文件 —— src/config/templates/config_template.yml（首次运行要拿它生成
       config.yml）、src/cli/help_commands.json（--help 的元数据）。用
       --include-package-data=src 一次带上。
    2. 动态导入的模块 —— ruamel.yaml（按名字插件式导入，漏了会回退 yaml.dump，
       config.yml 的注释全丢）；rich 13.x 的 rich._unicode_data（按 unicode17-0-0
       这种带版本号的名字动态导入）。后者按实际安装探测，14.x 不需要。
    3. 运行时要读的目录 —— webui dist（WebUI 静态资源）、extensions\（内置扩展，
       启动时扫描发现）、balcon\ 与 blb2txt\（扩展调用的第三方引擎 exe）。

    版本号与时间戳自动取值，避免手改命令时版本漂移。

.PARAMETER Clean
    编译前清掉 compiled_output。

.PARAMETER Minimal
    不带 balcon/blb2txt 引擎（省 13MB）。对应扩展仍可见，真正调用时才提示缺 exe，
    可用 PPC_BALCON / PPC_BLB2TXT 环境变量指向自己放置的引擎。

.PARAMETER OneFile
    打包成单个 exe（启动稍慢）。

.PARAMETER NoLto
    关闭 LTO（编译快很多，产物略慢）。

.PARAMETER Jobs
    编译并发数，默认 CPU 逻辑核心数。

.PARAMETER DryRun
    只打印将执行的命令，不编译。

.EXAMPLE
    .\scripts\build_exe.ps1 -Clean
    完整打包（带 WebUI、扩展、balcon/blb2txt）。

.EXAMPLE
    .\scripts\build_exe.ps1 -Minimal -NoLto -Jobs 8
    轻量打包，跳过大引擎与 LTO，用于快速验证。
#>
[CmdletBinding()]
param(
    [switch]$Clean,
    [switch]$Minimal,
    [switch]$OneFile,
    [switch]$NoLto,
    [int]$Jobs = 0,
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Entry       = Join-Path $ProjectRoot 'ppc.py'
$Icon        = Join-Path $ProjectRoot 'icon.ico'
$OutputDir   = Join-Path $ProjectRoot 'compiled_output'

function Get-ProjectVersion {
    $versionFile = Join-Path $ProjectRoot 'src\version.py'
    if (-not (Test-Path $versionFile)) { throw "找不到 $versionFile" }
    $match = Select-String -Path $versionFile -Pattern '__version__\s*=\s*["'']([^"'']+)["'']'
    if (-not $match) { throw '无法从 src\version.py 解析 __version__' }
    return $match.Matches[0].Groups[1].Value
}

function ConvertTo-WindowsVersion([string]$Version) {
    # Windows 版本资源要求四段数字：12.0.1 → 12.0.1.0
    # tag 风格（12.0.1-rc2-pre.1）的预发布标记在文件属性里没有位置，直接丢弃
    if ($Version -match '^(\d+)\.(\d+)\.(\d+)') {
        return "$($Matches[1]).$($Matches[2]).$($Matches[3]).0"
    }
    return '0.0.0.0'
}

function Test-PythonModule([string]$Module) {
    & $Python -c "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('$Module') else 1)" 2>$null
    return ($LASTEXITCODE -eq 0)
}

# ---------------------------------------------------------------- 前置检查

$Python = if ($env:PYTHON) { $env:PYTHON } else { 'python' }
try { & $Python --version *> $null } catch { throw "找不到 Python（$Python）。可设置环境变量 PYTHON 指定解释器。" }

$nuitkaVersion = & $Python -m nuitka --version 2>&1 | Select-Object -First 1
if ($LASTEXITCODE -ne 0) { throw '未安装 Nuitka：pip install nuitka' }

if (-not (Test-Path $Entry)) { throw "找不到入口 $Entry" }
if (-not (Test-Path $Icon)) { Write-Warning "缺少 icon.ico，将不带图标编译（不影响运行）" }

$webuiDist = Join-Path $ProjectRoot 'webui\WebUI-Vue\dist'
if (-not (Test-Path $webuiDist)) {
    Write-Warning ('webui\WebUI-Vue\dist 不存在：打出来的包 WebUI 不好用。' +
        '先在 webui\WebUI-Vue 下 npm ci && npm run build（本地库 WinUI-Web-Library 要先各自构建）')
}

$hasClang = $null -ne (Get-Command clang -ErrorAction SilentlyContinue)
if (-not $hasClang) { Write-Warning 'PATH 中没有 clang，将回退 MSVC（若也没有则 Nuitka 会下载 MinGW）' }

$version    = Get-ProjectVersion
$winVersion = ConvertTo-WindowsVersion $version
$stamp      = Get-Date -Format 'yyyyMMddHH'
$jobCount   = if ($Jobs -gt 0) { $Jobs } else { [Environment]::ProcessorCount }

Write-Host "版本: $version  ->  Windows 版本资源 $winVersion"
Write-Host "Nuitka: $nuitkaVersion"
Write-Host "并发: $jobCount"

# ---------------------------------------------------------------- 清理

if ($Clean -and (Test-Path $OutputDir)) {
    Write-Host "清理旧产物 $OutputDir"
    Remove-Item -LiteralPath $OutputDir -Recurse -Force -ErrorAction SilentlyContinue
}

# ---------------------------------------------------------------- 组装参数

# 必需：目录 → dist 内目标路径
$RequiredDirs = @(
    @{ Src = 'webui/WebUI-Vue/dist'; Dest = 'webui/WebUI-Vue/dist'; Why = 'WebUI 静态资源（缺则 /webui 404）' }
    @{ Src = 'extensions';           Dest = 'extensions';           Why = '内置扩展源码（缺则三个扩展消失）' }
)

# 可选：第三方引擎，只有对应扩展真正被调用时才需要
$OptionalDirs = @(
    @{ Src = 'balcon';  Dest = 'balcon';  Why = 'balcon TTS 后端引擎' }
    @{ Src = 'blb2txt'; Dest = 'blb2txt'; Why = '电子书抽取引擎' }
)

$argList = [System.Collections.Generic.List[string]]::new()
$argList.Add('-m')
$argList.Add('nuitka')
$argList.Add('--standalone')
$argList.Add("--output-dir=$OutputDir")
$argList.Add('--output-filename=ppc.exe')
# 首次编译 Nuitka 会下载 dependency walker / ccache 等，交互确认会卡住
$argList.Add('--assume-yes-for-downloads')
if ($OneFile) { $argList.Add('--onefile') }
$argList.Add("--jobs=$jobCount")
$argList.Add('--include-package-data=src')
$argList.Add('--include-package=ruamel.yaml')
# 测试与脚本不该进产物
$argList.Add('--nofollow-import-to=tests,scripts,compiled_output')
$argList.Add('--windows-console-mode=force')

# rich 13.x 把字符宽度表拆成 unicode<ver>.py 再拼名字动态导入，静态分析看不到；
# 14.x 已改回静态的 _cell_widths.py，此时该包不存在，硬写会让 Nuitka 直接 FATAL。
if (Test-PythonModule 'rich._unicode_data') {
    $argList.Add('--include-package=rich._unicode_data')
    Write-Host 'rich: 检测到 _unicode_data（13.x），已显式包含'
}

$dirs = @($RequiredDirs)
if (-not $Minimal) { $dirs += $OptionalDirs }
foreach ($item in $dirs) {
    $srcPath = Join-Path $ProjectRoot ($item.Src -replace '/', '\')
    if (Test-Path $srcPath) {
        $argList.Add("--include-data-dir=$($item.Src)=$($item.Dest)")
    } else {
        Write-Warning "跳过 $($item.Src)：目录不存在（$($item.Why)）"
    }
}

$argList.Add('--company-name=BLY Team / BarbaterLI')
$argList.Add('--product-name=PPC_NEXT')
$argList.Add("--file-version=$winVersion")
$argList.Add("--product-version=$winVersion")
$argList.Add("--file-description=PPC $version release$stamp")
$argList.Add('--copyright=(c) BLY Team - github.com/BarbaterLI/PPC')
if (Test-Path $Icon) { $argList.Add("--windows-icon-from-ico=$Icon") }
if (-not $NoLto) { $argList.Add('--lto=yes') }
if ($hasClang) { $argList.Add('--clang') }
$argList.Add($Entry)

$printable = ($argList | ForEach-Object { if ($_ -match '\s') { '"' + $_ + '"' } else { $_ } }) -join ' '
Write-Host ''
Write-Host "$Python $printable"
Write-Host ''

if ($DryRun) { Write-Host ''; Write-Host '(-DryRun：未编译)'; return }

# ---------------------------------------------------------------- 编译

Push-Location $ProjectRoot
try {
    & $Python @argList
    $exitCode = $LASTEXITCODE
} finally {
    Pop-Location
}

if ($exitCode -ne 0) {
    Write-Error "编译失败（exit=$exitCode）"
    exit $exitCode
}

$target = if ($OneFile) { Join-Path $OutputDir 'ppc.exe' } else { Join-Path $OutputDir 'ppc.dist' }
Write-Host ''
Write-Host "产物: $target"
if (Test-Path $target -PathType Container) {
    $sizeMB = [math]::Round(((Get-ChildItem $target -Recurse -File | Measure-Object Length -Sum).Sum / 1MB), 1)
    Write-Host "大小: $sizeMB MB"
}
Write-Host ''
Write-Host '自检：'
Write-Host "  cd $target"
Write-Host '  .\ppc.exe --version'
Write-Host '  .\ppc.exe doctor --no-network'
