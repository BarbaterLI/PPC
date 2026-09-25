# PPC_NEXT

**BarbaterLI 项目开发组 (BLY Team) 出品 — 批量文本转语音 (TTS) 有声书生产工具**

PPC_NEXT 将大批量 `.txt` 章节文件转换为高音质 MP3 有声书：内置 Edge TTS 云端合成
（含 UA/MUID 轮换、熔断、限流、静默重试等风控自适应机制）、章节分割、断点续传、
失败隔离、多节点分布式执行、插件扩展体系与 Vue WebUI。

当前版本见 [`src/version.py`](src/version.py)，变更历史见 [CHANGELOG.md](CHANGELOG.md)。

---

## 特性总览

| 领域 | 能力 |
|---|---|
| 合成 | Edge TTS（内置定制副本 + 环境包双后端）、多语音回退、语速/音量控制、字级时间戳 (SRT/WordBoundary) |
| 管线 | 智能分段（标点驱动）、并发工作池、渐进预热 (ramp-up)、段级缓存断点重建 (`resume`)、隔离区自动重试 |
| 成品 | 章节内嵌 ID3v2 CHAP/CTTS + 封面/元数据的单文件有声书 (`--book`)、MP3 帧级零解码合并 |
| 可靠 | 熔断器、指数退避重试、NoAudioReceived 静默重试、自适应超时 (fixed/auto/adaptive/synthetic)、检查点原子写 |
| 分布式 | master/node 模式、健康检查、负载均衡、任务迁移 |
| 扩展 | `.ppcext` 包管理、静态安全扫描、沙箱 (受限文件系统/网络/socket)、IPC (命名管道/socket)、executable 扩展 |
| 界面 | Typer CLI（`ppc help` 交互式帮助浏览器）、Flask + Vue3 WebUI（SSE 实时进度）、JSON 结构化输出 |

## 安装

要求 Python ≥ 3.12（Windows / Linux / macOS）。

```bash
pip install -r requirements.txt
# 可选 extras：
#   pip install "ppc-next[extra]"    # pydub/lxml 等兼容后端依赖（v12 起合并不再必需）
#   pip install "ppc-next[distributed]"
```

源码直跑：`python ppc.py --help`。Nuitka standalone 打包后 `ppc.exe` 同路径运行。

## 快速开始

```bash
# 1. 编辑 config.yml（voice/并发/分段等，全部带注释）
# 2. 批量转换目录内全部 .txt：
ppc convert ./novel_chapters ./audio
# 3. 单文件模式：
ppc convert 第1章.txt --one
# 4. 中断后续传：
ppc convert ./novel_chapters ./audio --resume
# 5. 整本成品（章节标记 + 元数据 + 封面）：
ppc convert ./novel_chapters ./book.mp3 --book --title "夜航船" --author "张岱" --cover cover.jpg
# 6. WebUI：
ppc webui            # http://127.0.0.1:8000
# 7. 环境自诊断：
ppc doctor
```

常用子命令：`convert` / `split`（章节分割）/ `batch`（批量归档）/ `resume` /
`voices` / `config` / `ext`（扩展管理）/ `dist`（分布式）/ `webui` / `help` / `bench`。
完整参数用 `ppc <command> --help` 查看。

## 配置

- 单一配置文件 `config.yml`，注释随写回保留（ruamel 往返序列化）。
- 预设：`--preset speed|balanced|quality|custom`。
- 旧版本配置自动迁移（`src/config/migration.py`），升级大版本时给出迁移摘要。
- 环境变量 `PPC_API_TOKEN` / `--api-token` 为 WebUI `/api/*` 启用 Bearer 鉴权。

## 扩展体系

`extensions/` 目录下的 `.ppcext`（zip）包可注册 CLI 子命令、文本/音频处理钩子、
TTS 后端（`ppc.tts.v1` 入口点协议）与 WebUI 页面。`ppc ext create` 生成骨架，
安装时执行静态安全扫描 + 权限沙箱。番茄小说下载器 (`fanqie`) 为内置示例扩展。

## 开发与测试

```bash
pip install -r requirements.txt
python -m pytest            # 全量离线套件（integration 默认跳过，PPC_RUN_INTEGRATION=1 启用）
python -m pytest tests/smoke # CLI 冒烟
ppc bench                   # 性能基线：冷启动 / 合并 / IO 单遍 三元组
python -m ruff check src    # lint
python -m mypy src          # 类型检查
```

仓库治理：`src/version.py` 是版本号唯一来源（有守护测试）；破坏性变更写入 CHANGELOG。

## 许可

见 [LICENSE](LICENSE)。
