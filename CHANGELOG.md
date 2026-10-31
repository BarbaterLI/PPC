# Changelog

本文件记录 PPC_NEXT 的版本变更。版本号的唯一来源为 `src/version.py`。

## [12.0.1-rc2-pre.1] - 2026-10

多角色配音优化：**按角色池合并（同一角色的多句话合批成一次请求）+ 按字幕时间戳切回**，
以及 **LLM 自动分配发声人**。全部默认关闭，不改变既有行为。

### 新增
- **AI 引号级标注（`tts.annotate` / `ppc convert --annotate`）**：用 OpenAI 兼容
  模型判定「每个引号是谁说的」，比正则推断准得多（实测 20 章 633 个引号、1 次
  调用）；失败整章回退正则匹配器，绝不阻断转换。
  - AI 只输出**游程** `[[起,止,"角色名"],…]`，不复述正文：数组格式实测漏 19/727
    项会导致后续整体错位，游程可校验「无缝覆盖 0..N-1」。
  - 结果按章节文本指纹缓存，重跑零成本；多章合并成一次请求
    （`max_input_chars` 30 万字 / `max_output_chars` 6 万字），超出自动切批。
  - `cast.aliases` 提供「角色 → 别名」表（如 太白金星 → 李长），只喂给模型做
    归一，**不改正文一个字**；`marker_policy` 控制标记密度（all / changes）。
  - 纯标点引号（如 `“……”`）会并入相邻轮次：实测边缘 TTS 对无可发音字符的文本
    返回 `NoAudioReceived`，会让整章合成失败。
- **标注失败降级到单章**：整批（最多 30 万字）标注不合格时自动拆成单章重试，
  避免一章的 JSON 问题拖累整批；新增 `annotate.max_calls` 预算上限（0=不限），
  超出后剩余章节回退正则匹配。
- **未入表角色汇总**：标注结束后打印 `其他:*` 的前 8 个与次数，提示补进 cast.yml
  （它们统一落到 `default` 音色，听感一致但统计上会拆成多个角色）。
- **`--annotate-only`**：只把缓存里的标注渲染成 `<章节>.annotated.txt`（`[voice:角色名]`
  标记，密度由 `marker_policy` 控制），供人工校对「这段到底谁读」，不合成音频。
- **标注缓存随角色表失效**：缓存键加入「角色名 + 别名」指纹——改 cast 后旧标注作废并
  重标，避免已删除的角色被 `profile_for` 静默降级到 `default` 音色（台词不丢但整条线
  换了音色，且无告警）。
- **归属语剥离（`tts.strip_attribution` / `--strip-attribution`）**：剥掉
  「李明说道：」这类纯归属语，旁白不再念；`mode=verb` 额外修剪句尾言说动词并
  保留动作描写。只动旁白轮次，台词永不触碰。
- **`--keep-segment-cache`**：合并后保留本章段缓存（便于断点续传/排查；默认清理）。
  清理改为逐文件删除，避免部分环境把「一次删几十个文件」当批量删除而中止进程。
- **角色级语速/音调/音量（cast.yml）**：角色可写成 `{voice, rate, pitch, volume}`，
  其中 rate/pitch/volume 是**相对全局的偏移**：最终值 = `tts.rate`（或 `--rate`）
  + 角色偏移，再夹到合法范围（±100% / ±100Hz）。用偏移而非绝对值，保证
  「整本 +10%」这类全局调整仍然生效、角色间相对快慢不丢。
  - 新增 `tts.pitch` 全局配置（此前 pitch 在客户端被硬编码 `+0Hz`，无法调整）；
    `pitch` 现已贯通 `config → 引擎 → client → Communicate`。
  - 合并分组键升级为 `(音色, 语速, 音调, 音量)`：同音色不同韵律的角色不会被
    合进同一批请求（否则一次请求只有一个 prosody，必然串音）。
  - 缓存键与段级 Journal 指纹纳入 pitch（仅非默认值时参与）：改音调即重新合成，
    且旧缓存不因升级集体失效。
  - 校验严格：`rate: 快`、拼错键名（`speed:`）、缺 `voice` 均直接报错并指出位置。
- **按角色池合并 + 字幕时间截断（`tts.merge_cut` / `ppc convert --merge-cut`）**：
  把**同一角色的全部片段汇成一个池**（``merge_scope="pool"``，默认），池内按
  `max_merge_pieces`/`max_merge_chars`/`max_merge_bytes` 分批，**每批一次普通
  TTS 请求**；合成后按 `SentenceBoundary` 时间戳帧对齐切回各片段，再按**原始
  片段下标**排序拼接，原文顺序由此还原。
  - 池内片段**无需在原文中相邻**——这正是「一个角色的多句话合并到一个请求」；
    排序靠 `_seg_XXX.mp3` 的下标，`SegmentJournal` / 断点续传 / 缓存语义不变。
  - `merge_scope="adjacent"` 为保守回退：只合并原文相邻的同音色片段。
  - **不使用 SSML、不跨角色合并**（实测服务端拒绝单 SSML 多 `<voice>`：只回
    `turn.start` 后断流）。
  - 判据按**连接数**而非字节阈值：传输层每 4096 字节分片各开一条 WebSocket，
    只有 `ceil((a+b)/4096) < ceil(a/4096)+ceil(b/4096)` 时才合并，避免
    「合并不省反而丢缓存粒度」。
  - **单批大小是可配置的硬约束**：`max_merge_chars`（默认 **1300 字**，即单次
    请求硬上限 ≈4096 字节）、`max_merge_bytes`（默认 4000）、`max_merge_pieces`
    （默认 64，仅作安全阀）。此前 `max_merge_chars=12000` 远超传输层上限，
    实际由片段数在卡批：实测 50k 字 / 1231 片段，按 8 段一批需 **158 次**请求，
    按 1300 字一批只需 **46 次**，逼近 `ceil(总字节/4096)=37` 的理论下限。
  - 四个兜底：句级对齐漂移（去标点逐字比对，失败**回退逐片段**）、首段前导静音
    （首片从 0 切）、叠帧 50ms（统一用下一片起点作上一片终点）、批内 all-or-nothing
    （`max_merge_pieces` 默认 8）。
  - 实测：真实联网 6 段 / 2 音色 → **2 请求（省 67%）**，字幕顺序与原文逐句一致；
    《女娲成长日记》前 20 章 / 591 段 / 4 音色离线测算——请求 **591 → 98（省 83%）**，
    `max_pieces=4` 时 171（71%）、`16` 时 66（89%）；对照 `adjacent` 模式为 0%
    （`split_speaker_turns` 已合并同说话人相邻轮次，相邻轮次必然换音色）。
- **字幕输出（`--subtitles` / `tts.merge_cut.subtitle_output`）**：接 `SubMaker`，
  按最终拼接时间轴（含段间静音）落 `.srt`。
  - 字幕模式下**自动关闭标点规范化**（默认会把 `。！？` 换成半角 `.!?`，服务端
    据此不再切句，实测 3240 字只给 1 条 cue；关闭后 124 条逐句）。
  - 相邻 cue 的 50ms 重叠自动裁剪到下一条起点，避免字幕叠字。
  - 需要 boundary 的合成会**跳过缓存读取**（缓存只复制音频、不产时间轴），
    保证每次开启字幕都能拿到真实时间戳。
- **LLM 自动分配发声人（`tts.llm_assign` / `ppc convert --llm-assign`）**：用
  **openai SDK**（OpenAI 兼容端点）读正文产出「角色→音色」表，运行时注入 `Cast`；
  `--save-cast cast.yml` 可落盘复用。
  - 硬约束：只接受 `candidate_voices` 池内音色（防编造）、只登记正文真实出现的角色
    （防臆造）、`main_roles` 钉死不被覆盖、`temperature=0`、严格 JSON。
  - 失败（未装库/无 key/网络/额度/JSON 非法）**只警告并降级**为现有音色，不中断转换。
- 配置项：`tts.merge_cut`（8 项，含 `merge_scope`）、`tts.llm_assign`（12 项），
  见 `config_template.yml`。

### 修复
- `client_base.synthesize_to_file` 新增可选 `boundary_sink`：`WordBoundary` /
  `SentenceBoundary` 元数据不再被丢弃（此前只写 `chunk.data`，字幕与切分数据
  全丢）。不传该参数时行为与旧实现完全一致。
- LLM 分配会**继承已给 cast 的角色**：LLM 未观察到的角色不再凭空消失
  （否则 `--cast cast.yml` 与 `--llm-assign` 同用时角色被清空、退化成旁白）。
- LLM 分配结果落盘失败时退化到系统临时目录，避免输出目录只读时白跑一次 LLM。

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
- 多角色配音 **`cast.yml`**：`ppc convert --cast cast.yml`（或配置 `tts.cast_file`）。
  **说话人轮次级**（非分段级）：每次说话人切换即一个独立合成单元，旁白/narrator 与每个
  角色各自音色、逐句切换；长轮次内部再按 `max_segment_length` 切块但音色不变。识别模式
  覆盖 5 种引号定界符（`“”「」『』""‘’`）+ 前向标签（`李明：`/`李明笑道：`）+ 后向归因
  （`”，李明笑道`）+ 破折号（`——李明：`）+ 冒号直述（`李明：…`）+ 独立成行 + 代词承接
  （`，他说`→继承）+ 连续引号继承；角色名最长匹配 + CJK 负向后顾（`小李` 不误命中 `李`）；
  相邻同说话人合并降请求数；`narrator`/`default` 兜底，缓存键含音色互不串音。
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
