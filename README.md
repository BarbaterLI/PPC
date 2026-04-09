# PPC8 - 冰璃岩文本转语音工具

<div align="center">


![Version](https://img.shields.io/badge/version-8.1.0-blue.svg)
![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)
![Python](https://img.shields.io/badge/python-3.8+-red.svg)
![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey.svg)

**冰璃岩项目开发组 (BLY Team) 出品**

一款功能强大、高性能的文本转语音批量处理工具，支持单机与分布式部署

[特性](#主要特性) | [快速开始](#快速开始) | [使用文档](#使用文档) | [分布式部署](#分布式部署) | [配置说明](#配置说明) | [架构设计](#架构设计) | [贡献指南](#贡献指南)

</div>

---

## 项目简介

PPC8 是一款专业的文本转语音 (TTS) 批量处理工具，专为小说、文档等长篇文本的语音转换而设计。基于 edge-tts (Azure TTS) 引擎，支持多种语音模型、智能章节分割、高并发处理、完善的错误恢复机制，以及 v8.1.0 新增的分布式多节点协同工作能力，是制作有声书、语音读物的理想选择。

### 核心优势

- **高效批量处理** - 支持高并发 TTS 转换，充分利用系统资源
- **智能章节分割** - 自动识别小说章节结构，智能分割文本
- **企业级可靠性** - 熔断器、重试机制、隔离队列，确保任务完成
- **分布式架构** - 多节点协同工作，支持负载均衡与故障转移
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
- 音频播放速度调节（-100% 到 +100%）

### 章节分割

- 智能识别中文/英文小说章节标题
- 支持自定义分割规则（正则表达式）
- 最小章节长度保护，避免碎片化
- 多种编码格式自动检测（UTF-8、GBK 等）
- 文件名长度自动优化

### 分布式部署 (v8.1.0 新增)

- **主控端-节点模式** - 支持多节点协同工作，提升整体吞吐量
- **负载均衡** - 轮询、最少连接、最快响应三种策略可选
- **健康检查** - 自动检测节点状态，故障节点自动剔除
- **故障转移** - 节点故障时自动切换到其他节点或本地执行
- **动态扩展** - 支持运行时动态添加/移除节点
- **独立并发控制** - 主控端和节点各自配置并发数

### 可靠性保障

- **熔断器模式** - 快速失败保护，避免雪崩效应，支持 CLOSED/OPEN/HALF_OPEN 三态转换、慢调用检测、窗口失败率计算
- **指数退避重试** - 智能重试策略，提高成功率
- **隔离队列** - 失败任务隔离，延迟重试
- **速率限制** - 自适应速率控制，防止 API 限流
- **错误分类** - 详细的错误类型和修复建议
- **断点续传** - 任务中断后可从检查点恢复

### 性能优化

- **连接池** - 复用网络连接，减少握手开销，支持预热、自适应扩缩容、健康检查
- **内存池** - 分代内存管理，降低 GC 压力
- **多级缓存** - L1 内存缓存 (LRU) + L2 磁盘缓存，支持 TTL 和模式失效
- **内存监控** - 实时内存使用监控，避免 OOM
- **性能分析** - 内置性能分析器，定位瓶颈
- **动态超时** - 基于历史记录的自适应超时计算

### 用户体验

- **Rich CLI** - 美观的命令行界面，色彩丰富
- **实时进度** - 并行进度条，任务状态一目了然
- **交互式帮助** - 内置帮助浏览器，命令查询方便
- **配置向导** - 交互式配置，新手友好
- **详细统计** - 转换完成后生成详细报告
- **实时监控** - 系统状态监控仪表板

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
pip install edge-tts rich pydantic psutil pydub aiohttp
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

# 从上次中断处继续（断点续传）
ppc8 convert ./input_texts ./output_audios --resume

# 调整音频播放速度
ppc8 convert ./input_texts ./output_audios --rate +10%
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
ppc8 config wizard

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

# 实时监控模式
ppc8 status --watch
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

| 参数                       | 说明                          | 默认值                         |
| -------------------------- | ----------------------------- | ------------------------------ |
| `INPUT_DIR`                | 输入目录（包含文本文件）      | 必需                           |
| `OUTPUT_DIR`               | 输出目录（保存音频文件）      | 必需                           |
| `-v, --voice`              | 语音模型 ID                   | 使用配置文件值                 |
| `-c, --concurrency`        | 并发数 (1-64)                 | 使用配置文件值                 |
| `-p, --preset`             | 配置预设                      | `balanced`                     |
| `-r, --resume`             | 从上次中断处继续              | `False`                        |
| `--checkpoint`             | 检查点文件路径                | 输出目录/.ppc8_checkpoint.json |
| `-t, --timeout-multiplier` | 超时倍率 (0.5-2.0)            | 使用配置文件值                 |
| `--rate`                   | 音频播放速度（如 +10%, -10%） | `+0%`                          |

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

| 参数                    | 说明                     | 默认值          |
| ----------------------- | ------------------------ | --------------- |
| `INPUT_FILE`            | 输入文件路径             | 必需            |
| `-o, --output`          | 输出目录                 | 当前目录        |
| `-p, --preset`          | 分割预设                 | `chinese_novel` |
| `-r, --custom-rules`    | 自定义规则 JSON          | 无              |
| `--add-title-separator` | 是否在章节名后添加分隔符 | 使用配置文件值  |

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

| 参数               | 说明     | 默认值  |
| ------------------ | -------- | ------- |
| `SOURCE_DIR`       | 源目录   | 必需    |
| `-b, --batch-size` | 批次大小 | `100`   |
| `--dry-run`        | 预览模式 | `False` |

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
- `init` - 初始化配置文件
- `path` - 显示配置文件路径
- `wizard` - 启动交互式配置向导

#### `ppc8 check` - 系统检查

检查系统环境和依赖。

**用法：**

```bash
ppc8 check [OPTIONS]
```

**参数：**

| 参数           | 说明                     | 默认值  |
| -------------- | ------------------------ | ------- |
| `-f, --full`   | 完整检查（包含网络）     | `False` |
| `-e, --export` | 导出检查结果为 JSON 文件 | 无      |

#### `ppc8 voices` - 列出语音

显示所有可用的 TTS 语音模型。

#### `ppc8 status` - 系统状态

显示系统资源使用情况和配置状态。

**参数：**

| 参数          | 说明         | 默认值  |
| ------------- | ------------ | ------- |
| `-w, --watch` | 实时监控模式 | `False` |

### 分布式命令

#### `ppc8 dist node` - 启动节点服务

启动 TTS 节点服务，接收主控端分配的任务。

**用法：**

```bash
ppc8 dist node [OPTIONS]
```

**参数：**

| 参数                | 说明           | 默认值    |
| ------------------- | -------------- | --------- |
| `-h, --host`        | 节点监听地址   | `0.0.0.0` |
| `-p, --port`        | 节点监听端口   | `8000`    |
| `-c, --concurrency` | 节点最大并发数 | `4`       |
| `-C, --config`      | 配置文件路径   | 无        |

#### `ppc8 dist status` - 查看分布式状态

查看分布式系统状态信息。

**参数：**

| 参数           | 说明            | 默认值 |
| -------------- | --------------- | ------ |
| `-C, --config` | 配置文件路径    | 无     |
| `-e, --export` | 导出状态为 JSON | 无     |

#### `ppc8 dist add-node` - 添加节点

添加分布式节点到配置中。

**参数：**

| 参数                | 说明               | 默认值   |
| ------------------- | ------------------ | -------- |
| `HOST`              | 节点 IP 地址       | 必需     |
| `PORT`              | 节点端口           | 必需     |
| `-c, --concurrency` | 节点最大并发数     | `4`      |
| `-C, --config`      | 配置文件路径       | 无       |
| `--save/--no-save`  | 是否保存到配置文件 | `--save` |

### 全局选项

| 选项             | 说明               |
| ---------------- | ------------------ |
| `-v, --verbose`  | 详细输出模式       |
| `--version`      | 显示版本号         |
| `--help`         | 显示帮助信息       |
| `--legacy`, `-l` | 使用 PPC2 遗留模式 |

---

## 分布式部署

### 架构设计

```
+----------------------------------------------------+
|                   主控端 (Master)                   |
|                                                     |
|  +----------------------------------------------+   |
|  |  任务调度器 (MasterScheduler)                  |   |
|  |  - 任务分配                                    |   |
|  |  - 负载均衡                                    |   |
|  |  - 故障转移                                    |   |
|  +----------------------------------------------+   |
|                       |                             |
|  +----------------------------------------------+   |
|  |  节点池 (NodePool)                             |   |
|  |  - 节点管理                                    |   |
|  |  - 健康检查                                    |   |
|  |  - 统计监控                                    |   |
|  +----------------------------------------------+   |
|                       |                             |
+----------------------------------------------------+
        |                 |                 |
   +---------+      +---------+      +---------+
   |节点 A    |      |节点 B    |      |节点 C    |
   |192.168  |      |192.168  |      |192.168  |
   |.1.100   |      |.1.101   |      |.1.102   |
   |:8000    |      |:8000    |      |:8000    |
   +---------+      +---------+      +---------+
```

### 快速开始

#### 1. 启动节点服务

在远程机器上启动节点：

```bash
# 在 192.168.1.100 上启动节点
ppc8 dist node --host 0.0.0.0 --port 8000 --concurrency 4

# 在 192.168.1.101 上启动节点
ppc8 dist node --host 0.0.0.0 --port 8000 --concurrency 8
```

#### 2. 配置主控端

在主控端添加节点：

```bash
# 添加第一个节点
ppc8 dist add-node 192.168.1.100 8000 --concurrency 4 --save

# 添加第二个节点
ppc8 dist add-node 192.168.1.101 8000 --concurrency 8 --save
```

#### 3. 查看分布式状态

```bash
ppc8 dist status
```

### 负载均衡策略

| 策略                 | 说明                                 | 适用场景       |
| -------------------- | ------------------------------------ | -------------- |
| `round_robin`        | 轮询，选择请求数最少的节点           | 通用场景       |
| `least_connections`  | 最少连接，选择当前并发最低的节点     | 节点性能差异大 |
| `best_response_time` | 最快响应，选择平均响应时间最短的节点 | 追求最低延迟   |

### 使用场景

#### 场景 1: 大规模批量转换

```bash
# 1. 在多台机器启动节点
# 机器 A
ppc8 dist node -p 8000 -c 8

# 机器 B
ppc8 dist node -p 8000 -c 8

# 机器 C
ppc8 dist node -p 8000 -c 8

# 2. 在主控端添加所有节点
ppc8 dist add-node 192.168.1.100 8000 -c 8 --save
ppc8 dist add-node 192.168.1.101 8000 -c 8 --save
ppc8 dist add-node 192.168.1.102 8000 -c 8 --save

# 3. 执行批量转换（自动使用分布式）
ppc8 convert ./input ./output -c 24
```

#### 场景 2: 本地+远程混合

```yaml
# config.yaml
distributed:
  enabled: true
  local_execution: true  # 主控端也参与执行
  nodes:
    - host: "192.168.1.100"
      port: 8000
      max_concurrency: 4
```

### 节点 API

节点服务提供 RESTful API：

**执行 TTS 合成:**

```http
POST /api/v1/synthesize
Content-Type: application/json

{
  "text": "你好，世界",
  "voice": "zh-CN-YunxiNeural",
  "rate": "+0%"
}
```

**健康检查:**

```http
GET /api/v1/health
```

**查看统计:**

```http
GET /api/v1/stats
```

### 最佳实践

- 节点并发建议 4-8，过高可能触发 Edge TTS 风控
- 主控端总并发 = 节点数 * 节点并发
- 节点性能相近：使用 `round_robin`
- 节点性能差异大：使用 `least_connections`
- 启用 `local_execution: true` 作为回退
- 确保主控端能访问所有节点的 IP:Port，延迟 < 100ms

---

## 配置说明

### 配置结构

PPC8 使用分层配置结构，所有配置项按功能模块组织：

```yaml
version: "8.1.0"
core:
  mode: parametric
  log_level: info
  temp_dir: ~/.cache/ppc7

tts:
  preset: balanced
  voice: zh-CN-YunxiNeural
  concurrency: 10
  retries: 3
  timeout_mode: auto
  rate: "+0%"
  rate_limit: 100
  max_segment_length: 5000
  segment_silence_ms: 100
  text_normalization:
    enable_whitespace_normalization: true
    enable_punctuation_normalization: true
    enable_trim_whitespace: true

split:
  preset: chinese_novel
  min_chapter_length: 100
  encoding_fallback: [utf-8, gbk, gb2312]

performance:
  memory_limit_mb: 768
  enable_memory_monitor: true
  enable_connection_pool: true
  connection_pool_size: 16

reliability:
  tts_retry:
    max_retries: 3
    base_delay: 2.0
    max_delay: 30.0
    exponential_base: 2.0
    jitter: 0.1
  tts_circuit:
    failure_threshold: 5
    success_threshold: 3
    timeout_seconds: 60.0
    half_open_max_calls: 3

distributed:
  enabled: false
  mode: master
  node_host: "0.0.0.0"
  node_port: 8000
  node_max_concurrency: 4
  nodes: []
  master_max_concurrency: 8
  load_balance_strategy: round_robin
  health_check_interval: 30
  task_timeout: 300
  max_retries: 3
  local_execution: true
```

### 核心配置 (core)

| 配置项              | 类型   | 默认值          | 说明                                     |
| ------------------- | ------ | --------------- | ---------------------------------------- |
| `mode`              | string | `parametric`    | 运行模式：parametric / interactive       |
| `log_level`         | enum   | `info`          | 日志级别：debug / info / warning / error |
| `temp_dir`          | string | `~/.cache/ppc7` | 临时文件目录                             |
| `progress_interval` | int    | `10`            | 进度回调触发频率                         |

### TTS 配置 (tts)

| 配置项                | 类型   | 默认值              | 说明                                          |
| --------------------- | ------ | ------------------- | --------------------------------------------- |
| `preset`              | string | `balanced`          | 配置预设：speed / balanced / quality / custom |
| `voice`               | string | `zh-CN-YunxiNeural` | 语音模型 ID                                   |
| `concurrency`         | int    | `10`                | 并发数 (1-64)                                 |
| `retries`             | int    | `3`                 | 重试次数                                      |
| `timeout_mode`        | string | `auto`              | 超时模式：fixed / auto / adaptive             |
| `timeout_min`         | int    | `50`                | 最小超时时间 (秒)                             |
| `timeout_max`         | int    | `720`               | 最大超时时间 (秒)                             |
| `max_segment_length`  | int    | `5000`              | 最大分段长度                                  |
| `segment_silence_ms`  | int    | `100`               | 音频片段间静音时长 (毫秒)                     |
| `rate`                | string | `+0%`               | 音频播放速度 (-100% 到 +100%)                 |
| `rate_limit`          | int    | `100`               | 速率限制                                      |
| `enable_segmentation` | bool   | `true`              | 启用智能分段                                  |

### 分割配置 (split)

| 配置项                | 类型   | 默认值                 | 说明                     |
| --------------------- | ------ | ---------------------- | ------------------------ |
| `preset`              | string | `chinese_novel`        | 章节预设                 |
| `min_chapter_length`  | int    | `100`                  | 最小章节长度             |
| `encoding_fallback`   | list   | `[utf-8, gbk, gb2312]` | 编码回退列表             |
| `custom_rules`        | list   | `[]`                   | 自定义分割规则列表       |
| `add_title_separator` | bool   | `false`                | 是否在章节名后添加分隔符 |

### 性能配置 (performance)

| 配置项                   | 类型 | 默认值 | 说明           |
| ------------------------ | ---- | ------ | -------------- |
| `memory_limit_mb`        | int  | `768`  | 内存限制 (MB)  |
| `enable_memory_monitor`  | bool | `true` | 启用内存监控   |
| `enable_connection_pool` | bool | `true` | 启用连接池     |
| `connection_pool_size`   | int  | `16`   | 连接池大小     |
| `max_file_cache_size`    | int  | `100`  | 最大文件缓存数 |

### 可靠性配置 (reliability)

**重试策略 (tts_retry):**

| 配置项             | 类型  | 默认值 | 说明           |
| ------------------ | ----- | ------ | -------------- |
| `max_retries`      | int   | `3`    | 最大重试次数   |
| `base_delay`       | float | `2.0`  | 基础延迟 (秒)  |
| `max_delay`        | float | `30.0` | 最大延迟 (秒)  |
| `exponential_base` | float | `2.0`  | 指数退避基数   |
| `jitter`           | float | `0.1`  | 抖动范围 (0-1) |

**熔断器 (tts_circuit):**

| 配置项                | 类型  | 默认值 | 说明               |
| --------------------- | ----- | ------ | ------------------ |
| `failure_threshold`   | int   | `5`    | 失败次数阈值       |
| `success_threshold`   | int   | `3`    | 成功次数阈值       |
| `timeout_seconds`     | float | `60.0` | 熔断超时时间 (秒)  |
| `half_open_max_calls` | int   | `3`    | 半开状态最大调用数 |

### 分布式配置 (distributed)

| 配置项                   | 类型   | 默认值        | 说明                    |
| ------------------------ | ------ | ------------- | ----------------------- |
| `enabled`                | bool   | `false`       | 启用分布式模式          |
| `mode`                   | string | `master`      | 运行模式：master / node |
| `node_host`              | string | `0.0.0.0`     | 节点监听地址            |
| `node_port`              | int    | `8000`        | 节点监听端口            |
| `node_max_concurrency`   | int    | `4`           | 节点最大并发数          |
| `nodes`                  | list   | `[]`          | 远程节点列表            |
| `master_max_concurrency` | int    | `8`           | 主控端最大并发数        |
| `load_balance_strategy`  | string | `round_robin` | 负载均衡策略            |
| `health_check_interval`  | int    | `30`          | 健康检查间隔 (秒)       |
| `task_timeout`           | int    | `300`         | 任务超时 (秒)           |
| `max_retries`            | int    | `3`           | 任务最大重试次数        |
| `local_execution`        | bool   | `true`        | 主控端也执行任务        |

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
├── ppc8.py                     # 主入口
├── config.yaml                 # 配置文件
├── src/
│   ├── core/                   # 核心层 - 领域模型和基础类型
│   ├── config/                 # 配置管理
│   ├── engines/                # 引擎层 - 业务逻辑引擎
│   │   ├── tts_engine.py       # TTS 引擎
│   │   └── chapter_engine.py   # 章节引擎
│   ├── executors/              # 执行器层 - 任务编排
│   │   ├── tts.py              # TTS 执行器
│   │   ├── splitter.py         # 分割执行器
│   │   ├── batcher.py          # 批处理执行器
│   │   ├── checkpoint.py       # 检查点管理
│   │   └── quarantine.py       # 隔离队列
│   ├── services/               # 服务层 - 应用服务
│   ├── cli/                    # 表现层 - 命令行界面
│   │   ├── typer_app.py        # Typer 应用入口
│   │   └── commands/           # 命令实现
│   │       ├── convert.py
│   │       ├── split.py
│   │       ├── distributed.py
│   │       └── ...
│   ├── distributed/            # 分布式模块 (v8.1.0)
│   │   ├── node_server.py      # 节点服务
│   │   ├── node_pool.py        # 节点池
│   │   └── master_scheduler.py # 主控端调度器
│   ├── infrastructure/         # 基础设施层
│   ├── reliability/            # 可靠性模块
│   │   ├── result.py           # 执行结果类型
│   │   ├── retry.py            # 重试策略
│   │   ├── circuit.py          # 熔断器
│   │   └── errors.py           # 错误分类
│   ├── text/                   # 文本处理
│   ├── audio/                  # 音频处理
│   ├── cache/                  # 缓存系统
│   ├── pool/                   # 资源池
│   ├── timeout/                # 超时管理
│   ├── events/                 # 事件系统
│   ├── logging/                # 日志系统
│   ├── tracing/                # 链路追踪
│   ├── scheduler/              # 任务调度
│   ├── profiler/               # 性能分析
│   ├── utils/                  # 工具函数
│   └── legacy/                 # 遗留代码 (PPC2 兼容)
├── tests/                      # 测试用例
├── docs/                       # 文档
│   ├── 分布式TTS系统实现总结.md
│   └── 分布式TTS系统使用指南.md
├── output/                     # 输出目录
└── README.md                   # 本文件
```

### 设计模式

- **协议接口**: `TTSEngineProtocol` 定义节点接口
- **策略模式**: 负载均衡策略可插拔
- **观察者模式**: 事件回调系统
- **工厂模式**: 重试策略和熔断器创建
- **依赖注入**: TTSExecutor 支持外部注入依赖

---

## API 参考

### 核心模块

#### 公开 API (src/__init__.py)

```python
from src import __version__, ConfigManager, get_default_config_dir, TyperApp

print(__version__)  # "8.1.0"
```

#### TTS 引擎

```python
from src.engines.tts_engine import TTSEngine
from src.config import PPC8Config

config = PPC8Config()
engine = TTSEngine(config)

# 合成语音
result = await engine.synthesize("你好，世界", output_path)

# 分段合成
result = await engine.synthesize_segmented(long_text, output_path)
```

#### TTS 执行器

```python
from src.executors.tts import TTSExecutor
from src.config import PPC8Config

config = PPCConfig()

async with TTSExecutor(config) as executor:
    result = await executor.add_batch_with_progress(
        input_dir,
        output_dir,
        progress_handler=handler
    )
```

#### 熔断器

```python
from src.reliability.circuit import CircuitBreaker, create_tts_circuit_breaker

breaker = create_tts_circuit_breaker(
    failure_threshold=5,
    timeout_seconds=60
)

# 使用熔断器保护操作
result = await breaker.execute(risky_operation)
```

#### 重试策略

```python
from src.reliability.retry import create_tts_retry_policy, RetryPolicy

retry_policy = create_tts_retry_policy(
    max_retries=3,
    base_delay=2.0,
    max_delay=30.0
)
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
pip install edge-tts rich pydantic psutil pydub aiohttp
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

## 版本历史

### v8.1.0 (2026-04-08)

- 新增分布式 TTS 系统支持
- 新增节点服务和节点池管理
- 新增主控端任务调度器
- 新增负载均衡（轮询、最少连接、最快响应）
- 新增健康检查和故障转移
- 新增分布式 CLI 命令

### v8.0.0

- 全新重构的架构设计
- 新增超时管理模块
- 新增缓存系统
- 新增性能分析器
- 改进错误处理和重试机制
- 改进配置管理系统

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
