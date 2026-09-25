# Changelog

本文件记录 PPC_NEXT 的版本变更。版本号的唯一来源为 `src/version.py`。

## [Unreleased] — 12.0.0 开发中（alpha）

### 新增
- `ppc bench`：离线性能基线（冷启动 / 音频合并 / IO 遍数），作为性能回归门禁。
- 版本号单一来源守护测试（禁止版本字面量散落在源码/模板中）。

### 变更
- CLI 冒烟测试强制 UTF-8 解码子进程输出（修复中文 Windows 上 5 个既有失败）。
- `.gitignore` 扩展：忽略运行时 `.cache/` 与前端 `.eslintcache`；`.cache` 移出版本控制。
- 配置模板 `config_template.yml` 版本号改为 `{{version}}` 占位符渲染。

### 删除
- `tests/unit/extensions/supersplit/`（8 个模块）与 `tests/unit/extensions/cloud_gradio/`（1 个模块）：
  所测试的扩展早已被移除，导致整个测试套件无法收集（ModuleNotFoundError）。

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
- 定时调度器 `src/scheduler/task_scheduler.py`（cron，依赖可选 `croniter`）。
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
