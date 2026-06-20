# PPC10

> 终极文本转语音工具 · 把整本小说变成 MP3

![version](https://img.shields.io/badge/version-10.1.0-blue)
![python](https://img.shields.io/badge/python-%3E%3D3.10-green)
![license](https://img.shields.io/badge/license-MIT-lightgrey)
![tts](https://img.shields.io/badge/TTS-edge--tts-orange)

- **版本**: 10.1.0
- **作者**: 冰璃岩项目开发组 (BLY Team)
- **仓库**: <https://github.com/BarbaterLI/PPC>
- **TTS 引擎**: [edge-tts](https://github.com/rany2/edge-tts)（Microsoft Azure Neural TTS）

PPC10 是一款专业的文本转语音（TTS）与内容处理平台，能够把整本小说或书籍的 `.txt` 文件批量转换为高音质 MP3 音频。它基于 Microsoft Azure Neural TTS（通过 `edge-tts`），提供章节智能分割、批量归档、断点续传、分布式集群处理、WebUI 可视化操作以及可扩展的插件系统，适合个人听书、内容创作者以及需要大规模文本转语音的场景。

---

## 目录

- [项目简介](#项目简介)
- [核心特性](#核心特性)
- [架构总览](#架构总览)
- [环境要求](#环境要求)
- [安装](#安装)
- [快速开始](#快速开始)
- [CLI 命令手册](#cli-命令手册)
- [WebUI](#webui)
- [配置系统](#配置系统)
- [分布式](#分布式)
- [扩展系统](#扩展系统)
- [开发指南](#开发指南)
- [退出码](#退出码)
- [FAQ](#faq)
- [许可证](#许可证)

---

## 项目简介

PPC10 由 **冰璃岩项目开发组（BLY Team）** 出品，定位为「终极文本转语音工具」。它的核心能力是把文本小说 / 书籍批量合成为 MP3 音频，并围绕这一核心构建了一整套工程化能力：

- 使用 Microsoft Azure Neural TTS（通过 `edge-tts`）作为语音合成后端，支持上百种神经网络语音。
- 提供智能章节分割器，可按「第 X 章」「卷-章-体」等中文小说常见结构切分。
- 支持批量归档、断点续传、并发预热、限流熔断，长时间运行稳定可靠。
- 内置 master / worker 分布式架构，可横向扩展到多台机器。
- 提供 Flask + React + Fluent UI 的 WebUI，零命令行也能完成转换。
- 内置番茄小说下载扩展，可一键拉取在线小说并转 MP3。

无论你是想听完一本几百万字的网文，还是需要为内容平台批量生产音频，PPC10 都能在本地或集群上稳定地完成工作。

---

## 核心特性

- **批量 TTS 转换**：把目录下所有 `.txt` 一次性合成为 MP3，支持并发、超时自适应、分段合并。
- **智能章节分割**：内置 `chinese_novel` 等预设，识别「第 X 章 / 第 X 节 / 卷 X」结构；支持卷-章-体层级分割与自定义规则。
- **批量归档**：按批次大小或按卷把源文件分批归档，支持 `--dry-run` 预览。
- **断点续传**：`convert --resume` 从检查点继续；`resume` 命令可从 `.cache` 分段重建检查点。
- **分布式处理**：master / worker 架构，主控负责任务分发，worker 节点执行 TTS；支持负载均衡、故障迁移、本地兜底。
- **WebUI**：Flask 后端 + React + Fluent UI + Vite 前端，可视化发起转换、查看进度、管理配置与扩展。
- **扩展系统**：通过 `src/extensions/` 加载内置与第三方扩展，内置番茄小说下载器；扩展可注册 CLI 子命令与 Web API。
- **配置系统**：Pydantic + YAML，支持预设（`speed` / `balanced` / `quality`）、热重载、`config wizard` 交互式向导。
- **可靠性**：网络与 TTS 双熔断器、指数退避重试、限流（令牌桶）、并发预热（`--ramp-up`）规避风控。
- **系统分析与健康检查**：`analyze` 命令提供轻量健康检查与深度分析（性能 / 配置 / 错误 / 依赖 / 网络 / 资源 / 代码质量），支持 `--fix` 自动修复、`--watch` 持续监控、HTML 报告导出。
- **单文件模式**：`convert --one` 针对单本大文件无超时、无限重试，确保完成。
- **多格式输出**：默认 MP3，支持元数据嵌入、章节间静音、命名模板。
- **跨平台**：Windows / Linux / macOS 均可运行。

---

## 架构总览

PPC10 采用分层架构，从入口到底层支撑依次为：

```
┌──────────────────────────────────────────────────────────────┐
│  入口层  ppc10.py                                            │
│    ├── CLI  (Typer + Rich)        src/cli/typer_app.py       │
│    └── WebUI (Flask)              src/web/app.py             │
├──────────────────────────────────────────────────────────────┤
│  核心引擎层                                                   │
│    ├── engines/        TTS / 章节 / EPUB / HTML / MD / PDF   │
│    ├── executors/      splitter / batcher / tts / merger     │
│    ├── text/           normalizer / segmenter                │
│    └── audio/          processor / post_processor            │
├──────────────────────────────────────────────────────────────┤
│  支撑层                                                       │
│    ├── config/         Pydantic schema + YAML manager        │
│    ├── reliability/    circuit / retry / rate_limiter        │
│    ├── distributed/    master / worker / scheduler           │
│    ├── analysis/       health check + deep analyzers         │
│    ├── extensions/     loader / base / fanqie                │
│    ├── timeout/        adaptive timeout calculator           │
│    ├── cache/          multilevel cache                      │
│    ├── events/         event bus                            │
│    ├── scheduler/      task / cron scheduler                 │
│    ├── tracing/        tracer                                │
│    ├── profiler/       profiler                              │
│    └── utils/          files / paths / format / validation   │
└──────────────────────────────────────────────────────────────┘
```

### 关键模块

| 模块 | 路径 | 职责 |
|------|------|------|
| CLI | `src/cli/` | Typer 应用、命令实现、输出格式化、交互式帮助 |
| 引擎 | `src/engines/` | `edge_tts_client`、`tts_engine`、`chapter_engine`、`epub/html/markdown/pdf_engine` |
| 执行器 | `src/executors/` | `splitter`、`batcher`、`tts_executor`、`merger`、`checkpoint`、`quarantine` |
| 配置 | `src/config/` | `schema`（Pydantic）、`manager`、`presets`、`migration` |
| 可靠性 | `src/reliability/` | `circuit`（熔断器）、`retry`（重试）、`rate_limiter`（限流）、`execution` |
| 分布式 | `src/distributed/` | `node_server`、`node_pool`、`scheduler`、`executor_adapter` |
| Web | `src/web/` | Flask `app`、`api/`（analyze/config/distributed/extensions/tasks/...）、`task_queue` |
| 扩展 | `src/extensions/` | `loader`、`base`、`package`、内置 `fanqie/` |
| 分析 | `src/analysis/` | `engine`、`repair`、`history`、`html_report`、`analyzers/`（性能/配置/错误/依赖/网络/资源/代码质量等） |
| 文本 | `src/text/` | `normalizer`、`segmenter` |
| 超时 | `src/timeout/` | `calculator`（自适应超时）、`history` |

入口 `ppc10.py` 默认走 Typer CLI；传入 `--webui` 时启动 Flask WebUI。所有业务命令实现在 `src/cli/commands/` 下，由 `src/cli/typer_app.py` 统一注册。

---

## 环境要求

| 项目 | 要求 |
|------|------|
| Python | `>=3.10`（`pyproject.toml` 要求 `>=3.10`，mypy 以 `3.12` 为目标） |
| 操作系统 | Windows / Linux / macOS |
| 网络 | 需可访问 `azure.microsoft.com`（Edge TTS 服务） |
| FFmpeg | 可选，音频后处理（`pydub`）需要时使用 |
| Node.js | 仅在开发 WebUI 前端时需要（构建 `webui/dist`） |

> 在 Python `>=3.13` 上会自动安装 `audioop-lts` 以替代被移除的 `audioop` 标准模块。

---

## 安装

```bash
# 1. 克隆仓库
git clone https://github.com/BarbaterLI/PPC.git ppc10
cd ppc10

# 2. 创建虚拟环境（推荐 Python 3.10+）
python -m venv .venv

# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

# 3. 安装依赖
pip install -r requirements.txt
```

依赖清单（核心）见 `requirements.txt`：

| 类别 | 依赖 |
|------|------|
| Web 框架 | Flask, Flask-CORS |
| TTS 引擎 | edge-tts |
| 音频处理 | pydub |
| 系统监控 | psutil |
| 解析 | beautifulsoup4, lxml |
| CLI | typer, rich |
| 配置 | pydantic, PyYAML |
| 网络 | requests, packaging |

---

## 快速开始

### 1. 把整本小说切分为章节

```bash
# 默认使用 chinese_novel 预设，输出到 ./chapters
python ppc10.py split novel.txt

# 指定输出目录与预设，启用卷章体层级分割
python ppc10.py split novel.txt -o ./chapters -p chinese_novel -H --add-title-separator
```

### 2. 批量转换为 MP3

```bash
# 把 ./chapters 下所有 .txt 转换为 MP3，输出到 ./tts
python ppc10.py convert ./chapters ./tts

# 指定语音、并发、语速
python ppc10.py convert ./chapters ./tts --voice zh-CN-XiaoxiaoNeural -c 8 --rate +10%

# 启用断点续传 + 并发预热（规避风控）
python ppc10.py convert ./chapters ./tts --resume --ramp-up 30
```

### 3. 单文件模式（一本大书一次跑完）

```bash
python ppc10.py convert ./book.txt --one
```

### 4. 启动 WebUI

```bash
python ppc10.py --webui
# 默认监听 http://0.0.0.0:5000
```

---

## CLI 命令手册

入口：`python ppc10.py <command> [options]`（下文示例简写为 `ppc10 <command>`）。

### 公共选项

所有命令均支持以下公共开关（在子命令前传入）：

| 选项 | 说明 |
|------|------|
| `--verbose`, `-v` | 详细输出（追加 stack trace） |
| `--quiet`, `-q` | 静默模式（仅打印结果摘要与错误） |
| `--json` | 结构化 JSON 输出（脚本化场景） |
| `--no-color` | 关闭 ANSI 颜色 |
| `--no-emoji` | 使用 ASCII 图标替代 emoji |
| `--timestamps` | 为人类可读日志添加时间戳前缀 |
| `--version`, `-V` | 显示版本信息并退出 |
| `--strict` / `--no-strict` | 严格模式：将 warning 视作 error（空输入目录 → 退出码 2） |

### `ppc10 convert` — 批量 TTS 转换

把 `input` 下的 `.txt` 文件批量转换为 `.mp3`，输出到 `output`。

```bash
ppc10 convert <input> [output] [options]
```

| 选项 | 说明 |
|------|------|
| `--voice` | 语音模型（默认使用配置文件） |
| `--concurrency`, `-c` | 并发数（默认使用配置文件） |
| `--preset`, `-p` | 配置预设（默认 `balanced`） |
| `--resume`, `-r` | 从上次中断处继续（断点续传） |
| `--checkpoint` | 检查点文件路径（默认 `output/.ppc10_checkpoint.json`） |
| `--timeout-multiplier`, `-t` | 超时倍率（0.5–2.0） |
| `--timeout-mode` | 超时模式 `fixed\|auto\|adaptive` |
| `--timeout` | 固定超时（秒），仅 `timeout_mode=fixed` 生效 |
| `--rate` | 音频播放速度（如 `+10%`、`-10%`，范围 -100% 到 +100%） |
| `--recursive`, `-R` | 递归处理子目录，保持目录结构 |
| `--ramp-up` | 并发预热时间（秒），规避风控 |
| `--one` | 单文件模式：单次无超时、无限重试 |

示例：

```bash
ppc10 convert ./txt ./out
ppc10 convert ./txt ./out --voice zh-CN-XiaoxiaoNeural -c 8
ppc10 convert ./txt ./out --rate +10% -r -t 1.5
ppc10 convert ./txt ./out --resume --ramp-up 30
ppc10 convert ./book.txt --one
```

### `ppc10 resume` — 重建断点续传检查点

从已有的 `.cache` 分段重建检查点，再用 `convert --resume` 继续。

```bash
ppc10 resume <input_dir> <output_dir> [options]
```

| 选项 | 说明 |
|------|------|
| `--voice` | 语音模型（默认使用配置文件） |
| `--checkpoint` | 检查点文件路径 |

示例：

```bash
ppc10 resume ./txt ./out
ppc10 resume ./txt ./out --voice zh-CN-XiaoxiaoNeural
ppc10 resume ./txt ./out --checkpoint ./ckpt.json
```

### `ppc10 split` — 章节分割

按预设或自定义规则把整本小说切分为若干章节。

```bash
ppc10 split <input_file> [options]
```

| 选项 | 说明 |
|------|------|
| `--output`, `-o` | 输出目录（默认 `chapters`） |
| `--preset`, `-p` | 章节预设（默认 `chinese_novel`） |
| `--custom-rules`, `-r` | 自定义规则 JSON 字符串或文件路径 |
| `--add-title-separator` / `--no-add-title-separator` | 是否在章节名后添加等于号分隔符 |
| `--hierarchical`, `-H` | 启用卷章体层级分割 |

示例：

```bash
ppc10 split novel.txt
ppc10 split novel.txt -o ./chapters -p chinese_novel
ppc10 split novel.txt -H --add-title-separator
ppc10 split novel.txt -r rules.json
```

### `ppc10 batch` — 批量归档

按批次或按卷把源目录中的 `.txt` 分批归档，支持预览。

```bash
ppc10 batch <source_dir> [options]
```

| 选项 | 说明 |
|------|------|
| `--batch-size`, `-b` | 每批次文件数 |
| `--dry-run`, `-d` | 预览模式 |
| `--group-by-volume`, `-g` | 按卷归档 |

示例：

```bash
ppc10 batch ./txt -b 50
ppc10 batch ./txt --dry-run --group-by-volume
ppc10 batch ./txt -g -b 100
```

### `ppc10 config` — 配置管理

```bash
ppc10 config <action> [options]
```

可用操作：`show` / `get` / `set` / `reset` / `export` / `import` / `init` / `path` / `wizard`。

| 选项 | 说明 |
|------|------|
| `--key`, `-k` | 配置键 |
| `--value`, `-v` | 配置值 |
| `--preset`, `-p` | 预设 |
| `--temp` | 临时设置 |
| `--export`, `-e` | 导出路径 |
| `--import`, `-i` | 导入路径 |
| `--full`, `-f` | 完整配置模式（wizard 专用） |

示例：

```bash
ppc10 config show
ppc10 config get --key tts.voice
ppc10 config set --key tts.voice --value zh-CN-XiaoxiaoNeural
ppc10 config reset --preset aggressive
ppc10 config wizard
```

### `ppc10 voices` — 列出可用语音

列出 Edge TTS 提供的所有可用语音，中文语音优先。

```bash
ppc10 voices
ppc10 voices --json
ppc10 voices --json | jq '.[0]'
```

### `ppc10 analyze` — 系统分析与健康检查

默认运行轻量级健康检查（系统环境、依赖、网络、文件系统、资源、配置验证）；`--deep` 启用深度分析。

```bash
ppc10 analyze [options]
```

| 选项 | 说明 |
|------|------|
| `--deep` | 启用深度分析 |
| `--performance`, `-p` | 性能分析 |
| `--config`, `-c` | 配置冲突检测 |
| `--errors`, `-e` | 错误模式识别 |
| `--dependency` | 依赖分析 |
| `--network` | 网络分析 |
| `--resource` | 资源分析 |
| `--quality` | 代码质量分析 |
| `--fix`, `-f` | 自动修复（交互式确认） |
| `--export`, `-x` | 导出分析报告为 JSON |
| `--diff` | 与最近历史对比 |
| `--watch`, `-w` | 持续监控模式 |
| `--interval`, `-i` | 监控间隔（秒，默认 60） |
| `--export-html` | 导出 HTML 报告 |
| `--full` | 完整检查（健康检查模式专用） |

示例：

```bash
ppc10 analyze                       # 健康检查
ppc10 analyze --fix                 # 健康检查并尝试一键修复
ppc10 analyze --export report.json  # 导出健康检查结果
ppc10 analyze --deep                # 深度分析
ppc10 analyze --deep --performance  # 仅性能深度分析
ppc10 analyze --deep --watch -i 30  # 每 30 秒持续监控
ppc10 analyze --deep --export-html r.html
```

### `ppc10 dist` — 分布式命令

分布式节点管理子命令组，包含 `node` / `master` / `status` / `add-node` / `convert`。

#### `ppc10 dist node` — 启动 TTS 节点服务

```bash
ppc10 dist node [options]
```

| 选项 | 说明 |
|------|------|
| `--host`, `-h` | 节点监听地址（默认 `0.0.0.0`） |
| `--port`, `-p` | 节点监听端口（默认 `8000`） |
| `--concurrency`, `-c` | 节点最大并发数（默认 `4`） |
| `--config`, `-C` | 配置文件路径 |

示例：

```bash
ppc10 dist node
ppc10 dist node --host 0.0.0.0 --port 8000 -c 4
ppc10 dist node -p 9001 -C /path/to/config.yml
```

#### `ppc10 dist master` — 启动分布式主控

```bash
ppc10 dist master [options]
```

| 选项 | 说明 |
|------|------|
| `--host`, `-h` | 主控监听地址（默认 `0.0.0.0`） |
| `--port`, `-p` | 主控监听端口（默认 `9000`） |
| `--concurrency`, `-c` | 主控最大并发数（仅本地兜底使用） |
| `--config`, `-C` | 配置文件路径 |
| `--add-worker` | 注册 worker 地址，可多次传入 |
| `--local-fallback` | 无 worker 时本地兜底执行 TTS |

示例：

```bash
ppc10 dist master
ppc10 dist master --port 9000
ppc10 dist master --add-worker 10.0.0.1:8000 --add-worker 10.0.0.2:8000
ppc10 dist master --local-fallback
```

#### `ppc10 dist status` — 查看分布式状态

```bash
ppc10 dist status [options]
```

| 选项 | 说明 |
|------|------|
| `--config`, `-C` | 配置文件路径 |
| `--export`, `-e` | 导出状态为 JSON |
| `--human` | 人类可读表格输出（默认 JSON） |
| `--json` | 结构化 JSON 输出（默认即 JSON） |
| `--quiet`, `-q` | 静默模式 |
| `--verbose`, `-v` | 详细输出 |

示例：

```bash
ppc10 dist status            # 默认 JSON
ppc10 dist status --human    # 表格输出
ppc10 dist status --export status.json
```

#### `ppc10 dist add-node` — 添加分布式节点

```bash
ppc10 dist add-node <host> <port> [options]
```

| 选项 | 说明 |
|------|------|
| `--concurrency`, `-c` | 节点最大并发数 |
| `--config`, `-C` | 配置文件路径 |
| `--save` / `--no-save` | 是否保存到配置文件 |

示例：

```bash
ppc10 dist add-node 10.0.0.1 8000
ppc10 dist add-node 10.0.0.1 8000 -c 8 --no-save
```

#### `ppc10 dist convert` — 提交任务到远端主控

等价于 `ppc10 convert` 的参数，但实际执行发生在主控端。

```bash
ppc10 dist convert <input_dir> <output_dir> [options]
```

| 选项 | 说明 |
|------|------|
| `--master`, `-m` | 主控端点（默认 `http://127.0.0.1:9000`） |
| `--config`, `-C` | 配置文件路径 |
| `--voice`, `-V` | 语音模型 |
| `--rate` | 音频播放速度 |
| `--concurrency`, `-c` | 并发数 |
| `--local-fallback` / `--no-local-fallback` | 无可用 worker 时本地兜底 |
| `--timeout` | HTTP 请求超时（秒，默认 3600） |

示例：

```bash
ppc10 dist convert ./txt ./out
ppc10 dist convert ./txt ./out --master http://10.0.0.1:9000
ppc10 dist convert ./txt ./out -V zh-CN-YunxiNeural -c 16
```

### `ppc10 docs` — 文档命令

文档子命令组，包含 `list` / `show` / `new` / `validate` / `spec`。

```bash
ppc10 docs list                 # 列出 docs/ 与 .trae/specs/ 下的 markdown 文档
ppc10 docs list --json
ppc10 docs show exit-codes      # 模糊匹配并渲染指定文档
ppc10 docs new my-new-doc       # 在 docs/dev/ 下创建带 frontmatter 模板
ppc10 docs validate             # 扫描所有 markdown，报告坏链接 / 越界锚点
ppc10 docs spec                 # 列出所有 spec
ppc10 docs spec mvp-cleanup     # 查看指定 spec 的任务与 checklist 完成度
```

### `ppc10 ext` — 扩展命令

扩展子命令组，自动加载 `src/extensions/` 下的扩展并注册其 CLI 子命令。内置 `fanqie`（番茄小说）扩展，可通过 `ppc10 ext fanqie ...` 调用。

### `ppc10 help` — 交互式帮助浏览器

```bash
ppc10 help
ppc10 --no-emoji help
```

进入交互式帮助浏览器，列出所有命令，支持上下导航、搜索、查看详情，按 `q` 或 `Ctrl+C` 退出。在 `--json` / `--quiet` 模式下直接输出命令索引 JSON。

---

## WebUI

PPC10 内置基于 Flask + React + Fluent UI + Vite 的 WebUI，提供可视化操作界面。

### 启动

```bash
python ppc10.py --webui
# 默认监听 http://0.0.0.0:5000

# 自定义 host / port / debug
python ppc10.py --webui --host 127.0.0.1 --port 8080 --debug
```

首次启动时，若 `webui/dist` 不存在但 `webui/` 存在，会自动执行 `npm run build` 构建前端；若未安装 Node.js，请先 `cd webui && npm install && npm run build`。

### 提供的能力

- 发起 TTS 转换、章节分割、批量归档任务并查看进度
- 配置管理（查看 / 编辑 / 切换预设）
- 分布式节点状态查看
- 扩展管理（含番茄小说下载）
- 系统健康检查与分析
- 任务队列与历史

后端 API 位于 `src/web/api/`，覆盖 `analyze` / `config` / `distributed` / `extensions` / `fanqie` / `operations` / `system` / `tasks` / `schema` 等模块。

---

## 配置系统

PPC10 使用 **Pydantic + YAML** 管理配置。配置文件默认为项目根目录的 `config.yml`。

### 配置文件位置

```bash
ppc10 config path     # 显示当前配置文件路径
ppc10 config init     # 初始化配置文件
```

### 主要配置段

| 段 | 说明 |
|------|------|
| `core` | 核心运行参数：`log_level`、`mode`、`progress_interval`、`temp_dir` |
| `tts` | TTS 引擎：`voice`、`concurrency`、`rate`、`timeout`、`timeout_mode`、分段、限流、文本归一化等 |
| `split` | 章节分割：`preset`、`min_chapter_length`、`hierarchical_split`、`add_title_separator`、编码回退 |
| `batch` | 批量归档：`max_files_per_batch`、`max_size_mb`、`preserve_order` |
| `output` | 输出：`default_format`、`audio_quality`、`output_naming`、`metadata_embed`、`silence_between_chapters_ms` |
| `distributed` | 分布式：`mode`、`nodes`、`load_balance_strategy`、`fault_tolerance`、`adaptive_load_balance` |
| `reliability` | 可靠性：`tts_retry`、`tts_circuit`、`network_retry`、`network_circuit`、`tts_no_audio` |
| `rate_limit` | 限流：`max_requests_per_second`、`burst_size`、`strategy`（`token_bucket`） |
| `network` | 网络探测：`probe_hosts`、`probe_interval`、`timeout` |
| `extensions` | 扩展：`enabled`、`auto_load`、`extension_dirs`、`installed_extensions` |
| `pipeline` | 流水线：`max_parallel_steps`、`pipeline_dirs`、`saved_pipelines` |
| `performance` | 性能：`memory_limit_mb`、`max_file_cache_size`、`stream_flush_threshold` |
| `features` | 功能开关：`auto_retry`、`keep_awake`、`merge_short_chapters`、`smart_detection` |
| `webhook` | Webhook：`enabled`、`url`、`events`、`secret`、`retry_count` |
| `ui` | UI 输出：`mode`、`verbose`、`no_color`、`show_progress`、`show_timestamps` |

### 预设

PPC10 提供开箱即用的预设，可通过 `--preset` 或 `config reset --preset` 切换：

| 预设 | 适用场景 |
|------|------|
| `speed` | 速度优先，高并发、低超时 |
| `balanced` | 平衡（默认），稳定与吞吐兼顾 |
| `quality` | 质量优先，低并发、高音质 |

### 配置向导

```bash
ppc10 config wizard          # 引导式配置
ppc10 config wizard --full   # 完整配置模式
```

### 配置示例

```yaml
tts:
  voice: zh-CN-YunxiNeural
  concurrency: 6
  rate: +0%
  timeout_mode: adaptive
  retries: 5
  enable_segmentation: true
  max_segment_length: 2000

split:
  preset: chinese_novel
  hierarchical_split: false
  min_chapter_length: 100

reliability:
  tts_retry:
    max_retries: 5
    base_delay: 1.0
    exponential_base: 2.0
  tts_circuit:
    failure_threshold: 5
    timeout_seconds: 60.0
```

---

## 分布式

PPC10 内置 master / worker 分布式架构，可把单台机器的批量 `convert` 任务派发到多台 worker 节点执行。

### 架构

```
┌────────────┐   convert 任务   ┌────────────┐
│  Client    │ ───────────────▶ │  Master    │
│ (dist conv)│                  │  (调度)    │
└────────────┘                  └─────┬──────┘
                                      │ 分发任务
                       ┌──────────────┼──────────────┐
                       ▼              ▼              ▼
                 ┌──────────┐  ┌──────────┐  ┌──────────┐
                 │ Worker 1 │  │ Worker 2 │  │ Worker N │
                 │ (TTS)    │  │ (TTS)    │  │ (TTS)    │
                 └──────────┘  └──────────┘  └──────────┘
```

- **Master**：仅负责调度与转发，监听默认端口 `9000`；可通过 `--add-worker` 注册 worker，或在无 worker 时 `--local-fallback` 本地兜底。
- **Worker（Node）**：实际执行 TTS 转换，监听默认端口 `8000`，通过 `--concurrency` 控制单节点并发。
- **故障容错**：`distributed.fault_tolerance` 支持任务迁移、降级、恢复检查；`adaptive_load_balance` 提供基于历史负载的自适应均衡。

### 典型流程

```bash
# 1. 在每台 worker 机器上启动节点
ppc10 dist node --host 0.0.0.0 --port 8000 -c 8

# 2. 在主控机器上启动 master 并注册 worker
ppc10 dist master --add-worker 10.0.0.1:8000 --add-worker 10.0.0.2:8000

# 3. 查看集群状态
ppc10 dist status --human

# 4. 提交转换任务到主控
ppc10 dist convert ./txt ./out --master http://10.0.0.1:9000 -c 16
```

也可通过 `ppc10 dist add-node <host> <port>` 动态添加节点（默认保存到配置文件）。

---

## 扩展系统

PPC10 通过 `src/extensions/` 提供插件机制，扩展可注册 CLI 子命令与 Web API。

### 工作机制

- `ExtensionLoader` 在启动时扫描 `extensions.extension_dirs`（默认 `extensions`）目录。
- 每个扩展继承 `extensions.base`，可声明 `metadata`（名称、描述）并实现 `register_cli(sub_app)` 注册 Typer 子命令。
- 扩展子命令统一挂载到 `ppc10 ext <name> ...` 下。
- 通过 `extensions.auto_load` 控制是否自动加载，`strict_validation` 控制校验严格程度。

### 内置扩展：番茄小说（fanqie）

位于 `src/extensions/fanqie/`，提供从番茄小说网站下载小说正文的能力，下载后可直接喂给 `convert` 转 MP3。

```bash
# 通过扩展命令调用（具体子命令取决于扩展注册的实现）
ppc10 ext fanqie ...
```

### 扩展示例

`src/extensions/examples/priority_lb.py` 提供了一个优先级负载均衡扩展示例，可作为开发第三方扩展的参考。

---

## 开发指南

### 项目脚本

| 脚本 | 用途 |
|------|------|
| `scripts/lint.py` | 运行 ruff lint |
| `scripts/typecheck.py` | 运行 mypy 类型检查 |
| `scripts/format.py` | 运行 ruff format 格式化 |
| `scripts/dev.ps1` | 开发模式启动（PowerShell） |
| `scripts/start.ps1` | 生产启动（PowerShell） |
| `scripts/backup_project.py` | 项目备份 |
| `scripts/check_ckpt.py` | 检查检查点文件 |
| `scripts/check_failed.py` | 检查失败任务 |
| `scripts/check_existing_audio.py` | 检查已存在音频 |

### 代码质量配置

`pyproject.toml` 中配置了 ruff 与 mypy：

```toml
[tool.ruff]
target-version = "py310"
line-length = 120

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "C4", "SIM", "N", "W"]

[tool.mypy]
python_version = "3.12"
files = ["src"]
ignore_missing_imports = true
warn_return_any = true
```

### 测试

测试位于 `tests/`，按 `unit` / `integration` / `smoke` 分类，使用 `pytest`：

```bash
pytest tests/
```

### 典型开发流程

```powershell
# 1. 格式化 + lint + 类型检查
python scripts/format.py
python scripts/lint.py
python scripts/typecheck.py

# 2. 运行测试
pytest tests/

# 3. 开发模式启动 WebUI
.\scripts\dev.ps1
```

### WebUI 前端开发

前端位于 `webui/`，技术栈为 React + Fluent UI + Vite：

```bash
cd webui
npm install
npm run dev      # 开发服务器
npm run build    # 构建到 webui/dist
```

---

## 退出码

PPC10 使用统一的退出码规约（详见 `docs/exit-codes.md`）：

| Code | 含义 | 示例 |
|------|------|------|
| 0 | 成功 / 正常工作 | 包括「无操作」场景 |
| 1 | 业务错误 | TTS 调用失败 / 文件 IO |
| 2 | 参数错误 | `E_INPUT_NOT_FOUND` / `E_INPUT_EMPTY` |
| 3 | 网络 / 外部依赖错误 | edge-tts 不可达 |
| 4 | 权限错误 | 无法写输出目录 |
| 5 | 配置错误 | `E_CONFIG_MISSING` / `E_CONFIG_INVALID` |

> 默认无操作场景退出码为 `0`；`--strict` 模式下退出码为 `2`。

---

## FAQ

### Q1：TTS 转换失败 / 报网络错误怎么办？

A：先运行 `ppc10 analyze` 做健康检查，确认能否访问 `azure.microsoft.com`。若偶发失败，PPC10 会按 `reliability.tts_retry` 自动重试；若持续失败，可降低并发（`-c`）、启用 `--ramp-up` 预热，或切换语音模型。深度排查使用 `ppc10 analyze --deep --network`。

### Q2：转换中途中断了，如何续传？

A：若之前已启用 `--resume`，直接再次运行 `ppc10 convert <input> <output> --resume` 即可从检查点继续。若之前未启用 `--resume` 但生成了 `.cache` 分段，先运行 `ppc10 resume <input> <output>` 重建检查点，再 `convert --resume`。

### Q3：如何更换语音？

A：用 `ppc10 voices` 查看所有可用语音（中文优先），然后通过命令行 `--voice zh-CN-XiaoxiaoNeural` 临时指定，或通过 `ppc10 config set --key tts.voice --value zh-CN-XiaoxiaoNeural` 永久写入配置。

### Q4：如何使用分布式加速？

A：在多台机器上分别 `ppc10 dist node` 启动 worker，然后在主控机 `ppc10 dist master --add-worker <host:port>` 注册它们，最后用 `ppc10 dist convert <input> <output> --master <master_url>` 提交任务。单机也可用 `--local-fallback` 让主控兜底执行。

### Q5：配置文件在哪里？如何重置？

A：运行 `ppc10 config path` 查看路径（默认项目根目录 `config.yml`）。重置使用 `ppc10 config reset --preset balanced`（可选 `speed` / `balanced` / `quality`）。新手推荐 `ppc10 config wizard` 交互式配置。

---

## 许可证

本项目使用 MIT 许可证，详见仓库根目录的 [LICENSE](LICENSE) 文件。

---

<p align="center">
  冰璃岩项目开发组 (BLY Team) · PPC10 v10.1.0<br>
  <a href="https://github.com/BarbaterLI/PPC">https://github.com/BarbaterLI/PPC</a>
</p>
