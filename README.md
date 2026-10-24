# PPC_NEXT

**BarbaterLI 项目开发组 (BLY Team) 出品 — 批量文本转语音 (TTS) 有声书生产工具**

PPC_NEXT 将大批量 `.txt` 章节文件转换为高音质 MP3 有声书：内置 Edge TTS 云端合成
（含 UA/MUID 轮换、熔断、限流、静默重试等风控自适应机制）、章节分割、断点续传、
失败隔离、多节点分布式执行、插件扩展体系与 Vue WebUI。

> **当前为 `12.0.0-rc1` 预发布版本**（版本号唯一来源 [`src/version.py`](src/version.py)，
> 变更历史见 [CHANGELOG.md](CHANGELOG.md)）。

---

## 特性总览

| 领域 | 能力 |
|---|---|
| 合成 | Edge TTS（内置定制副本 + 环境包双后端）、多语音回退、语速/音量控制、字级时间戳 (SRT/WordBoundary)、多角色配音 (`--cast cast.yml`)、`ppc.tts.v1` 后端协议（内置 `openai_http`） |
| 管线 | 智能分段（标点驱动）、并发工作池、渐进预热 (ramp-up)、段级缓存断点重建 (`resume`)、段级 Journal 崩溃续传、隔离区自动重试 |
| 成品 | 整书有声书 `ppc convert --book`（ID3v2.4 CTOC/CHAP 章节时间轴，可跳章）、MP3 帧级零解码合并（同构 MP3 快路由） |
| 可靠 | 熔断器、指数退避重试、NoAudioReceived 静默重试、自适应超时 (fixed/auto/adaptive/synthetic)、检查点原子写 |
| 分布式 | master/node 模式、健康检查、负载均衡、任务迁移 |
| 扩展 | `.ppcext` 包管理、静态安全扫描、沙箱 (受限文件系统/网络/socket)、IPC (命名管道/socket)、executable 扩展 |
| 运维 | `ppc doctor` 环境自诊断（14 项检查，异常隔离、失败退出码 1）、`ppc bench` 性能基线三元组、JSON 结构化输出 |
| 界面 | Typer CLI（`ppc help` 交互式帮助浏览器）、Flask + Vue3 WebUI（SSE 实时进度） |

## 安装

要求 Python ≥ 3.12（Windows / Linux / macOS）。

```bash
pip install -r requirements.txt
# 可选 extras：
#   pip install "ppc-next[audio]"     # pydub 音频后端（v12 起同构 MP3 合并不再必需）
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
# 5. 整本成品（章节标记 + 元数据，输出 <书名>.book.mp3）：
ppc convert ./novel_chapters ./audio --book --book-title "夜航船"
# 6. 多角色配音（说话人轮次级音色切换，narrator/角色兜底）：
ppc convert ./novel_chapters ./audio --cast cast.yml
# 7. 按角色池合并（同一角色的多句话合批成一次请求），再按字幕时间戳切回原位：
ppc convert ./novel_chapters ./audio --cast cast.yml --merge-cut
# 8. 顺带输出 .srt 字幕：
ppc convert ./novel_chapters ./audio --cast cast.yml --subtitles
# 9. LLM 自动分配发声人（OpenAI 兼容端点），结果落盘复用：
ppc convert ./novel_chapters ./audio --llm-assign \
    --candidate-voices zh-CN-YunxiNeural,zh-CN-XiaoxiaoNeural --save-cast cast.yml
# 10. WebUI：
ppc webui            # http://127.0.0.1:5000
# 11. 环境自诊断：
ppc doctor
```

常用子命令：`convert` / `split`（章节分割）/ `batch`（批量归档）/ `resume` /
`voices` / `config` / `ext`（扩展管理）/ `dist`（分布式）/ `webui` / `help` /
`bench` / `doctor`。完整参数用 `ppc <command> --help` 查看。

## 配置

- 单一配置文件 `config.yml`，注释随写回保留（ruamel 往返序列化）。
- 预设：`--preset speed|balanced|quality|custom`。
- 旧版本配置自动迁移（`src/config/migration.py`），升级大版本时给出迁移摘要。
- 环境变量 `PPC_API_TOKEN` / `--api-token` 为 WebUI `/api/*` 启用 Bearer 鉴权。
- 相关键：`tts.backend_plugin`（`ppc.tts.v1` 后端扩展）、`tts.cast_file`（多角色配音表）、
  `tts.merge_cut`（合并同角色片段 + 字幕时间截断）、`tts.llm_assign`（LLM 分配发声人）。

## 扩展体系

`extensions/` 目录下的 `.ppcext`（zip）包可注册 CLI 子命令、文本/音频处理钩子、
TTS 后端（`ppc.tts.v1` 入口点协议）与 WebUI 页面。`ppc ext create` 生成骨架，
安装时执行静态安全扫描 + 权限沙箱。番茄小说下载器 (`fanqie`) 为内置示例扩展。

插件可依赖的宿主接口经公共接缝 [`src/cli/extension_api.py`](src/cli/extension_api.py)
暴露（`console` / `ColorTokens` / `get_icon`）；插件**不得**反向导入宿主内部实现
（如 `src.cli.design.tokens`、`src.cli.output.console`），以保证宿主重构不破坏插件兼容性。

## 架构与依赖方向

模块间依赖保持单向，禁止双向包依赖环：

- **engines → extensions 单向**：`src/engines` 拥有 `ppc.tts.v1` 协议与后端注册表，
  不导入 `src.extensions`；扩展侧访问注册表时延迟导入该协议（内置 `openai_http`
  后端按协议鸭子类型实现，不继承扩展基类）。
- **extensions → engines 单向**：扩展加载器负责自动注册/反注册后端。
- **cli**：命令模块直连 `src.cli.output`；扩展经 `src/cli/extension_api.py` 公共接缝
  消费渲染能力。

## 开发与测试

```bash
pip install -r requirements.txt
python -m pytest            # 全量离线套件（integration 默认跳过，PPC_RUN_INTEGRATION=1 启用）
python -m pytest tests/smoke # CLI 冒烟
ppc bench                   # 性能基线：冷启动 / 合并 / IO 单遍 三元组
python -m ruff check src    # lint
python -m mypy src          # 类型检查
```

仓库治理：

- `src/version.py` 是版本号唯一来源（有守护测试）；
- CLI 帮助元数据 `src/cli/help_commands.json` 与 Typer 定义双向一致性由
  `tests/unit/cli/test_help_commands.py` 守护；
- 扩展 `extensions/fanqie/` 为唯一事实源，`fanqie.ppcext.zip` 由
  `scripts/build_ext_zip.py` 重建（单源守护测试）；
- 破坏性变更写入 CHANGELOG。

## 许可

见 [LICENSE](LICENSE)。