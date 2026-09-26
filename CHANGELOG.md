# Changelog

本文件记录 PPC_NEXT 的版本变更。版本号的唯一来源为 `src/version.py`。

## [12.0.0] - 2026-09

大版本：三线深度性能优化（启动 / IO / 合并）+ 五项新能力（整书、多角色配音、
后端协议、环境自诊断、段级 Journal 崩溃续传）+ 依赖精简。

### 性能
- 冷启动：import 模块 **380 → 222（−41%）**，import-self 显著下降；`ppc.py --version`
  快速路径在 Typer 导入前短路，版本/branding 走 `src/version.py` 单一来源，
  `src/utils`、`src/cli` 等 PEP 562 惰性导出 + `PYDANTIC_DISABLE_PLUGINS=1`。
  启动预算守护测试 `tests/unit/cli/test_startup_budget.py`（模块 ≤280、import-self ≤320ms）。
- 单遍 IO：`read_text_auto` 每文件整读 **1 次（原 2 次）** + `quick_probe` 仅读头部 1KB，
  `detect_encoding` 全程 **0 次**调用；批量语义不变（测试 `test_single_pass_io.py` 守护）。
- 音频合并：纯字节 MP3 帧探测（`src/audio/mp3_probe.py`）+ 零解码帧拼接（`mp3_concat.py`），
  同构 MP3 合并 60 段 pydub 28.7s → 快路由 0.16s（**≈180×**，随机器负载实测 180–690×）；
  单文件 / 全 MP3 路径无需解码。`pydub` / `audioop-lts` 由必需依赖降级为可选 extras `audio`。
- WorkerPool：限流门移到取任务**之前**，ramp-up 期间不再空转占用队列。

### 新增
- `ppc convert --book [--book-title]`：整本零解码装配为单文件有声书，写 **ID3v2.4
  CTOC + CHAP**（章节时间轴 + 内嵌标题，ffprobe / mpv / MusicBee 可读）+ 合成 Xing TOC
  双栈（第三方播放器生态覆盖）；产出 `输出目录/<书名>.book.mp3`，可跳章。
- 多角色配音 **`cast.yml`**：`ppc convert --cast cast.yml`（或配置 `tts.cast_file`）按
  分段文本判定角色换音色；三种识别模式（行首角色+冒号 / 说话动词归因 / 任意位置+冒号），
  `narrator`/`default` 兜底；缓存键已含音色，角色间互不串音。
- **`ppc.tts.v1` 后端协议**（`src/engines/tts_protocol.py`）：版本协商 + 能力声明
  （voices/probe）+ 惰性注册表；内置 **`openai_http`** 后端（OpenAI 兼容 `/audio/speech`，
  mp3 直通零解码管线，接 Kokoro/SoVITS/F5 等本地服务），扩展后端由加载器自动注册、
  卸载自动反注册；`voices` 枚举按后端能力路由。
- **`ppc doctor`** 环境自诊断：14 项独立检查（Python / 依赖 / edge-tts / pydub / ffmpeg /
  配置 / 可写目录 / `ppc.tts.v1` 后端 / 扩展清单 / 网络可达性），单项异常隔离不拖垮整体，
  失败退出码 1，`--no-network` 跳过联网、`--json` 结构化输出。
- **段级 Journal 崩溃续传**（`src/engines/segment_journal.py`）：`.cache/<stem>/journal.json`
  逐段记录 `(index, size, text-sha1)`，命中判断 **O(1)**（替代整段解码探针）；分段文本/
  配置变更哈希自动失效，杜绝「同名段内容已变却被复用」的错位拼接；11.x 旧缓存仍走解码探针
  复用并补登 journal；条目原子落盘（tmp + `os.replace`），有条目 ⟹ 段文件完整。
- `ppc bench`：离线性能基线（冷启动 / 音频合并 / IO 遍数）三元组，`--save` 落盘作前后对照、
  作为性能回归门禁。

### 变更
- 版本号 **11.0.0 → 12.0.0**（`src/version.py` 单一来源，守护测试通过；`pyproject.toml` /
  `config.yml` 同步）。
- 配置迁移 `migrate_v11_to_v12`（`VERSION_MAPPING["11.0"]`）：版本戳升 12.x、显式剥离
  12.x 移除的死段（`features`/`webhook`/`rate_limit`）、补 `_migrated_from` 元数据。
- 新增配置键 `tts.backend_plugin`、`tts.cast_file`（默认空，旧配置自动取默认值）；
  移除段由 pydantic 静默忽略，旧配置无缝升级。
- CLI 冒烟测试强制 UTF-8 解码子进程输出（修复中文 Windows 上 5 个既有失败）。
- `.gitignore` 扩展：忽略运行时 `.cache/` 与前端 `.eslintcache`；`.cache` 移出版本控制。
- 配置模板 `config_template.yml` 版本号改为 `{{version}}` 占位符渲染。
- 版本号单一来源守护测试（禁止版本字面量散落在源码/模板中）。

### 修复
- `--book` 的 CHAP 章节标记按 FFmpeg 实际写侧字节布局（`element_id` 后直接
  时间×2 + 偏移×2），修复第三方解析器读到 0 章节的问题。

### 删除
- `tests/unit/extensions/supersplit/`（8 个模块）与 `tests/unit/extensions/cloud_gradio/`（1 个模块）：
  所测试的扩展早已被移除，导致整个测试套件无法收集（ModuleNotFoundError）。
- 根目录 `fanqie/` 陈旧源码副本（运行时绝对导入已指向不存在的模块）；
  `extensions/fanqie/` 为唯一事实源，`fanqie.ppcext.zip` 改由
  `scripts/build_ext_zip.py` 从其重建。
- 零生产消费者的死模块：`src/tracing/`（823 行，无任何调用方）、
  `src/scheduler/`（359 行，无调用方）、`src/audio/post_processor.py`、
  `src/executors/registry.py`、`src/web/config.py`（PPC_WEB_* 环境变量半成品，
  CLI 参数已覆盖同能力），含对应测试。新增 `scripts/find_dead_modules.py` 守护。
- 依赖：`croniter`、`markdown`（源码从未导入）；`pydub`、`audioop-lts` 移入可选 extras `audio`。

## [11.0.0] - 2026-09

### 破坏性变更
- 版本号收敛到 `src/version.py` 单一来源；`config.yml` 顶层 `version` 与
  `ConfigManager.CONFIG_VERSION` 同步为 11.0.0。
- `tts.backend` 键移除（风控后端选择合并进 `tts.anti_risk_control.backend`）。
- 扩展入口统一：旧版顶层 `entry` 字段仍兼容读取，新清单使用 `entry_point`。
- Windows 提权重启路径重构为 `src/utils/elevation.py`。
- Python 最低版本要求提升为 3.12（Nuitka standalone 同）。

### 新增
- 番茄小说下载器（`fanqie`）：作为 `.ppcext` 内置扩展交付，
  CLI（`ppc ext fanqie ...`）与 WebUI 双入口，支持镜像源与 musl 静态构建偏好。
- 扩展体系升级：`package_type: executable` 原生扩展（命名管道 IPC + 资源监控 +
  sha256 校验），`.ppcext` 安装/卸载/列表/信息全链路（`ppc ext` 子命令、WebUI 扩展页）。
- 沙箱增强：受限文件系统/socket/subprocess 代理，AST 级静态安全扫描。
- 后台任务调度器 `src/scheduler/task_scheduler.py`（12.0.0 因无消费者而移除）。
- 分布式：HTTP master/node 服务、健康检查、自适应负载均衡与故障任务迁移。
- 可靠性：NoAudioReceived 专项静默重试、UA 池健康过滤与轮换、MUID 持久化、
  熔断半开冷却感知的重试退避、隔离区到期自动重入队。
- 自适应超时四模式 `fixed/auto/adaptive/synthetic`（历史 P95 + 启发式伸缩）。
- 并发渐进预热 `--ramp-up`（1 → N 并发，规避突发风控）。
- 断点续传体系：检查点 + `ppc resume` 从 `.cache` 段文件重建；段级缓存损坏探针。
- WebUI v2：Vue3 + TypeScript 重写（Dashboard/转换/切分/配置/扩展/分布式/番茄页面，
  SSE 实时进度，API Token 鉴权），`ppc webui` 自动构建缺失的 `dist`。
- CLI 体验：`--json` 结构化输出、`ppc help` 交互式帮助浏览器、版本卡片、错误码体系。
- 链路追踪 `src/tracing` 与多级缓存 `src/cache`（L1 内存 + L2 磁盘，TTL/驱逐）。
- 音频：MP3/WAV/OGG/AAC 输出、48k 比特率与 Edge 源对齐、响度标准化（可选 pyloudnorm）。
- EPUB 引擎与章节层级分割（卷/章/节）。
- Windows 权限错误自动 UAC 提权重启；跨平台 keep-awake。
- 配置写回保留注释（ruamel 往返序列化）。

### 修复
- `ppc resume` 不再丢失新增文件；隔离任务此前"只写不读"导致重试从未发生。
- 批处理下 `Session is closed`：TTS 引擎改为执行器级共享（避免每任务新建 aiohttp 会话）。
- `add_batch` 的 `recursive` 参数曾被静默丢弃。
- 文件级单次熔断 gate：修复单文件多段耗尽半开探针配额。

## [10.0.0] 及更早
- （历史版本变更未在本仓库保留，参见旧版发布说明）
