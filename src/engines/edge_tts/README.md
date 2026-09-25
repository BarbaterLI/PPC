# Vendored edge-tts（定制副本）

本目录 `src/engines/edge_tts/` 是 [edge-tts](https://github.com/rany2/edge-tts) 的**定制 fork**，由BarbaterLI项目开发组（BLY Team）复制并针对 PPC_NEXT 做了性能优化。它不是 pip 包的镜像，与上游存在差异，请勿直接覆盖为上游原版。

## 与 pip `edge-tts` 的关系

- **首选后端**：`src/engines/http_client.py` 优先从本目录导入（`from src.engines.edge_tts import ...`），后端名记为 `"src.engines.edge_tts (customized)"`。
- **回退后端**：若本目录导入失败，则回退到环境安装的 `edge-tts`（`backend=auto/env`），见 `http_client.py` 顶部 backend selection 段（L40-L74）。
- **依赖声明**：`requirements.txt` 中的 `edge-tts>=7.2.8` 用于提供回退后端及类型/常量参考，并非主要实现来源。

即：**仓库内定制副本是主实现，pip `edge-tts` 是兜底**。二者并存是刻意设计，来源并不重复。

## 已知定制点

- `communicate.py` — 模块级构建 `_INCOMPATIBLE_CHARS_TABLE` 翻译表，将服务不支持的字符范围一次性映射为空格。相比上游逐字符 `ord()` 判断 + 列表重建，`str.translate` 对长文本分段（默认 2500 字符）显著降低 CPU 开销。
- `communicate.py` — 内部方法为单下划线（`_stream` / `_parse_metadata` / `_compensate_offset`），`__stream` 已拆为 `_build_request_headers` / `_send_command_request` / `_send_ssml_request` / `_handle_text_message` / `_parse_audio_message`；内部状态为 `CommunicateState` dataclass（`typing.py`）；构造函数参数收敛为 `CommunicateConfig` dataclass，并支持注入 `DRM` 实例；已删除 `stream_sync` / `save_sync` 同步双套接口（无调用方）。
- `drm.py` — 时钟偏差（clock skew）由 `ClockSkew` 实例持有（`DRM.__init__(clock_skew=...)` 注入），不再使用类级可变属性 `clock_skew_seconds`；`generate_sec_ms_gec` / `handle_client_response_error` 等变为实例方法，共享默认实例 `DEFAULT_DRM` 保持进程内偏差持续生效。
- `data_classes.py` — `UtilArgs`（CLI 参数命名空间）移至 CLI 入口模块 `util.py`。

> 若后续新增定制，请在此处登记（文件名 + 改动点 + 原因），便于同步上游时核对。

## 与上游同步方法

1. 拉取上游发行版源码（如 `edge-tts==7.2.8`）到临时目录。
2. 逐文件与本目录对比，把本目录的定制改动（如上表）重新应用到新版本。
3. 运行 `tests/unit/engines/` 下相关测试，确认无回归。
4. 更新本 README 的定制点登记、版本与日期。

## 注意

- 本目录为源码副本，编译（Nuitka 等）打包时需包含 `src/engines/edge_tts/`。
- 不要在此目录直接提交上游变更；保持定制点集中、可审计。