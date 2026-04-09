# PPC8 - 冰璃岩文本转语音工具

<div align="center">

![Version](https://img.shields.io/badge/version-8.0.0-blue.svg)
![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)
![Python](https://img.shields.io/badge/python-3.8+-red.svg)
![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey.svg)

**冰璃岩项目开发组 (BLY Team) 出品**

一款功能强大、高性能的文本转语音批量处理工具

[特性](#主要特性) | [快速开始](#快速开始) | [使用文档](#使用文档) | [配置说明](#配置说明) | [架构设计](#架构设计) | [贡献指南](#贡献指南)

</div>

---

## 项目简介

PPC8 是一款专业的文本转语音 (TTS) 批量处理工具，专为小说、文档等长篇文本的语音转换而设计。基于 edge-tts (Azure TTS) 引擎，支持多种语音模型、智能章节分割、高并发处理、完善的错误恢复机制，是制作有声书、语音读物的理想选择。

### 核心优势

- **高效批量处理** - 支持高并发 TTS 转换，充分利用系统资源
- **智能章节分割** - 自动识别小说章节结构，智能分割文本
- **企业级可靠性** - 熔断器、重试机制、隔离队列，确保任务完成
- **灵活的配置系统** - 支持预设配置、自定义规则、配置导入导出
- **美观的 CLI 界面** - Rich 库打造的现代化命令行界面
- **完善的错误处理** - 详细的错误报告和修复建议

---

## 主要特性

### TTS 转换

- 支持多种 Azure TTS 语音模型（中文、英文等）
- 高并发处理，支持 1-64 并发数可调
- 智能文本分段，优化输出质量
- 自适应超时控制，避免任务卡死
- 速率限制和流量控制，防止 API 限流

### 章节分割

- 智能识别中文/英文小说章节标题
- 支持自定义分割规则（正则表达式）
- 最小章节长度保护，避免碎片化
- 多种编码格式自动检测（UTF-8、GBK 等）
- 文件名长度自动优化

### 可靠性保障

- **熔断器模式** - 快速失败保护，避免雪崩效应，支持 CLOSED/OPEN/HALF_OPEN 三态转换、慢调用检测、窗口失败率计算
- **指数退避重试** - 智能重试策略，提高成功率
- **隔离队列** - 失败任务隔离，延迟重试
- **速率限制** - 自适应速率控制，防止 API 限流
- **错误分类** - 详细的错误类型和修复建议

### 性能优化

- **连接池** - 复用网络连接，减少握手开销，支持预热、自适应扩缩容、健康检查
- **内存池** - 分代内存管理，降低 GC 压力
- **多级缓存** - L1 内存缓存 (LRU) + L2 磁盘缓存，支持 TTL 和模式失效
- **内存监控** - 实时内存使用监控，避免 OOM
- **性能分析** - 内置性能分析器，定位瓶颈

### 用户体验

- **Rich CLI** - 美观的命令行界面，色彩丰富
- **实时进度** - 并行进度条，任务状态一目了然
- **交互式帮助** - 内置帮助浏览器，命令查询方便
- **配置向导** - 交互式配置，新手友好
- **详细统计** - 转换完成后生成详细报告

---

## 快速开始

### 环境要求

- Python 3.8+
- Windows / Linux / macOS
- 网络连接（使用 Azure TTS 服务）
- ffmpeg（音频合并功能依赖，可选安装）

**安装 ffmpeg:**
```bash
# Windows (使用 choco)
choco install ffmpeg

# macOS (使用 brew)
brew install ffmpeg

# Ubuntu/Debian
sudo apt install ffmpeg

# CentOS/RHEL
sudo yum install ffmpeg
```

### 安装依赖

```bash
# 克隆项目
git clone https://github.com/BarbaterLI/PPC.git
cd PPC

# 安装依赖
pip install edge-tts rich pydantic psutil pydub
```

### 基础使用

#### 文本转语音转换

```bash
# 基本用法
ppc8 convert ./input_texts ./output_audios

# 指定语音模型和并发数
ppc8 convert ./input_texts ./output_audios -v zh-CN-YunxiNeural -c 16

# 使用质量优先预设
ppc8 convert ./input_texts ./output_audios --preset quality
```

#### 章节分割

```bash
# 使用默认预设分割小说
ppc8 split ./novel.txt

# 指定输出目录
ppc8 split ./novel.txt -o ./chapters

# 使用英文小说预设
ppc8 split ./english_novel.txt -p english_novel

# 自定义分割规则
ppc8 split ./novel.txt -r '{"name": "自定义", "pattern": "^第\\\\S+章"}'
```

#### 批量归档

```bash
# 批量归档文件
ppc8 batch ./source_dir -b 100

# 预览模式（不实际移动文件）
ppc8 batch ./source_dir --dry-run
```

#### 配置管理

```bash
# 显示当前配置
ppc8 config show

# 启动配置向导
ppc8 config --wizard

# 导出配置
ppc8 config export -e ./my_config.json

# 导入配置
ppc8 config import -i ./my_config.json

# 设置单个配置项
ppc8 config set -k tts.voice -v zh-CN-XiaoxiaoNeural
```

#### 系统检查

```bash
# 快速检查
ppc8 check

# 完整检查（包含网络探测）
ppc8 check --full

# 列出可用语音
ppc8 voices

# 显示系统状态
ppc8 status
```

---

## 使用文档

### 命令详解

#### `ppc8 convert` - TTS 转换

批量将文本文件转换为语音文件。

**用法：**
```bash
ppc8 convert [OPTIONS] INPUT_DIR OUTPUT_DIR
```

**参数：**
| 参数 | 说明 | 默认值 |
|------|------|--------|
| `INPUT_DIR` | 输入目录（包含文本文件） | 必需 |
| `OUTPUT_DIR` | 输出目录（保存音频文件） | 必需 |
| `-v, --voice` | 语音模型 ID | `zh-CN-YunxiNeural` |
| `-c, --concurrency` | 并发数 (1-64) | `8` |
| `-p, --preset` | 配置预设 | `balanced` |

**预设选项：**
- `speed` - 速度优先（低并发，快速超时）
- `balanced` - 平衡模式（中等并发，智能超时）
- `quality` - 质量优先（高并发，长超时）
- `custom` - 自定义配置

#### `ppc8 split` - 章节分割

将长文本按章节分割成多个文件。

**用法：**
```bash
ppc8 split [OPTIONS] INPUT_FILE
```

**参数：**
| 参数 | 说明 | 默认值 |
|------|------|--------|
| `INPUT_FILE` | 输入文件路径 | 必需 |
| `-o, --output` | 输出目录 | 当前目录 |
| `-p, --preset` | 分割预设 | `chinese_novel` |
| `-r, --custom-rules` | 自定义规则 JSON | 无 |

**预设选项：**
- `chinese_novel` - 中文小说（识别"第 X 章"格式）
- `english_novel` - 英文小说（识别"Chapter X"格式）
- `default` - 默认规则

#### `ppc8 batch` - 批量归档

将大量文件按批次归档。

**用法：**
```bash
ppc8 batch [OPTIONS] SOURCE_DIR
```

**参数：**
| 参数 | 说明 | 默认值 |
|------|------|--------|
| `SOURCE_DIR` | 源目录 | 必需 |
| `-b, --batch-size` | 批次大小 | `100` |
| `--dry-run` | 预览模式 | `False` |

#### `ppc8 config` - 配置管理

管理和配置 PPC8 参数。

**用法：**
```bash
ppc8 config ACTION [OPTIONS]
```

**操作类型：**
- `show` - 显示当前配置
- `get` - 获取单个配置项
- `set` - 设置配置项
- `reset` - 重置配置
- `export` - 导出配置
- `import` - 导入配置

#### `ppc8 check` - 系统检查

检查系统环境和依赖。

**用法：**
```bash
ppc8 check [OPTIONS]
```

**参数：**
| 参数 | 说明 | 默认值 |
|------|------|--------|
| `-f, --full` | 完整检查（包含网络） | `False` |

#### `ppc8 voices` - 列出语音

显示所有可用的 TTS 语音模型。

#### `ppc8 status` - 系统状态

显示系统资源使用情况和配置状态。

### 全局选项

| 选项 | 说明 |
|------|------|
| `-v, --verbose` | 详细输出模式 |
| `--version` | 显示版本号 |
| `--help` | 显示帮助信息 |
| `--legacy`, `-l` | 使用 PPC2 遗留模式 |

---

## 配置说明

### 配置结构

PPC8 使用分层配置结构，所有配置项按功能模块组织：

```yaml
version: "8.0.0"
core:
  mode: parametric
  log_level: info
  temp_dir: ~/.cache/ppc7

tts:
  preset: balanced
  voice: zh-CN-YunxiNeural
  concurrency: 8
  retries: 3
  timeout_mode: auto

split:
  preset: chinese_novel
  min_chapter_length: 100

performance:
  memory_limit_mb: 768
  enable_connection_pool: true

reliability:
  tts_retry:
    max_retries: 3
    exponential_base: 2.0
  tts_circuit:
    failure_threshold: 5
    timeout_seconds: 60
```

### 核心配置 (core)

| 配置项 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `mode` | string | `parametric` | 运行模式：parametric | interactive |
| `log_level` | enum | `info` | 日志级别：debug | info | warning | error |
| `temp_dir` | string | `~/.cache/ppc7` | 临时文件目录 |
| `progress_interval` | int | `10` | 进度回调触发频率 |

### TTS 配置 (tts)

| 配置项 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `preset` | string | `balanced` | 配置预设：speed | balanced | quality | custom |
| `voice` | string | `zh-CN-YunxiNeural` | 语音模型 ID |
| `concurrency` | int | `8` | 并发数 (1-64) |
| `retries` | int | `3` | 重试次数 |
| `timeout_mode` | string | `auto` | 超时模式：fixed | auto | adaptive |
| `timeout_min` | int | `50` | 最小超时时间 (秒) |
| `timeout_max` | int | `720` | 最大超时时间 (秒) |
| `max_segment_length` | int | `2500` | 最大分段长度 |
| `segment_silence_ms` | int | `100` | 音频片段间静音时长 (毫秒) |

### 分割配置 (split)

| 配置项 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `preset` | string | `chinese_novel` | 章节预设 |
| `min_chapter_length` | int | `100` | 最小章节长度 |
| `encoding_fallback` | list | `[utf-8, gbk, gb2312]` | 编码回退列表 |
| `custom_rules` | list | `[]` | 自定义分割规则列表 |

### 性能配置 (performance)

| 配置项 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `memory_limit_mb` | int | `768` | 内存限制 (MB) |
| `enable_memory_monitor` | bool | `true` | 启用内存监控 |
| `enable_connection_pool` | bool | `true` | 启用连接池 |
| `connection_pool_size` | int | `16` | 连接池大小 |
| `max_file_cache_size` | int | `100` | 最大文件缓存数 |

### 可靠性配置 (reliability)

**重试策略 (tts_retry):**
| 配置项 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `max_retries` | int | `3` | 最大重试次数 |
| `base_delay` | float | `2.0` | 基础延迟 (秒) |
| `max_delay` | float | `30.0` | 最大延迟 (秒) |
| `exponential_base` | float | `2.0` | 指数退避基数 |
| `jitter` | float | `0.1` | 抖动范围 (0-1) |

**熔断器 (tts_circuit):**
| 配置项 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `failure_threshold` | int | `5` | 失败次数阈值 |
| `success_threshold` | int | `3` | 成功次数阈值 |
| `timeout_seconds` | float | `60.0` | 熔断超时时间 (秒) |
| `half_open_max_calls` | int | `3` | 半开状态最大调用数 |

---

## 架构设计

PPC8 采用分层架构设计，包含以下层次：

```
表现层 (Presentation)
    |
服务层 (Services)
    |
执行器层 (Executors)
    |
引擎层 (Engines)
    |
核心层 (Core)
    |
基础设施层 (Infrastructure)
```

### 目录结构

```
PPC8/
├── ppc8.py                 # 主入口
├── config.yaml             # 配置文件
├── src/
│   ├── core/               # 核心层 - 领域模型和基础类型
│   ├── config/             # 配置管理
│   ├── engines/            # 引擎层 - 业务逻辑引擎
│   ├── engine/             # 引擎层 (旧，兼容性模块)
│   ├── executors/          # 执行器层 - 任务编排
│   ├── executor/           # 执行器层 (旧)
│   ├── services/           # 服务层 - 应用服务
│   ├── cli/                # 表现层 - 命令行界面
│   ├── infrastructure/     # 基础设施层
│   ├── reliability/        # 可靠性模块
│   ├── text/               # 文本处理
│   ├── audio/              # 音频处理
│   ├── cache/              # 缓存系统
│   ├── pool/               # 资源池
│   ├── timeout/            # 超时管理
│   ├── events/             # 事件系统
│   ├── logging/            # 日志系统
│   ├── tracing/            # 链路追踪
│   ├── scheduler/          # 任务调度
│   ├── profiler/           # 性能分析
│   ├── utils/              # 工具函数
│   └── legacy/             # 遗留代码
├── tests/                  # 测试用例
├── docs/                   # 文档
├── ARCHITECTURE_DESIGN.md  # 架构设计文档
└── README.md               # 本文件
```

详细的架构设计请参考 [ARCHITECTURE_DESIGN.md](ARCHITECTURE_DESIGN.md)。

---

## API 参考

### 核心模块

#### 公开 API (src/__init__.py)

```python
from src import __version__, ConfigManager, get_default_config_dir, TyperApp

print(__version__)  # "8.0.0"
```

#### TTS 引擎

```python
from src.engines.tts_engine import TTSEngine

engine = TTSEngine(
    voice="zh-CN-YunxiNeural",
    concurrency=8,
    retries=3
)

# 批量转换
results = engine.batch_convert(texts, output_dir)
```

#### 章节分割器

```python
from src.executor.splitter import ChapterSplitter

splitter = ChapterSplitter(preset="chinese_novel")

# 分割文件
chapters = splitter.split_file("novel.txt")

# 保存到目录
splitter.save_chapters(chapters, output_dir="./chapters")
```

#### 熔断器

```python
from src.reliability.circuit import CircuitBreaker

breaker = CircuitBreaker(
    failure_threshold=5,
    timeout_seconds=60
)

@breaker.protect
def risky_operation():
    # 可能失败的操作
    pass
```

---

## 贡献指南

### 开发环境设置

```bash
# 克隆项目
git clone https://github.com/BarbaterLI/PPC.git
cd PPC

# 创建虚拟环境
python -m venv venv
venv\Scripts\activate     # Windows
source venv/bin/activate  # Linux/macOS

# 安装开发依赖
pip install edge-tts rich pydantic psutil pydub
pip install pytest pytest-cov black flake8
```

### 代码规范

- 遵循 PEP 8 代码风格
- 使用中文注释和文档字符串
- 函数和方法添加类型注解
- 提交前运行测试和代码检查

```bash
# 代码格式化
black src/ tests/

# 代码检查
flake8 src/ tests/

# 运行测试
pytest tests/ -v
```

### 提交 PR

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

---

## 问题反馈

如有问题或建议，请通过以下方式联系：

- GitHub Issues: https://github.com/BarbaterLI/PPC/issues
- 仓库地址: https://github.com/BarbaterLI/PPC

---

## 致谢

感谢所有为 PPC 项目做出贡献的开发者和用户！

**冰璃岩项目开发组 (BLY Team)** - 致力于提供高质量的文本处理工具

---

## 许可证

本项目采用 Apache License 2.0 开源协议。

Copyright 2026 BLY Team

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

完整协议请查看项目中的 [LICENSE](LICENSE) 文件。

---

<div align="center">

**Made with by BLY Team**

</div>
