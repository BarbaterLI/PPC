"""扩展相关枚举定义。

包含扩展类型、扩展权限与安全级别枚举。
"""

from enum import Enum


class ExtensionType(Enum):
    """扩展类型枚举。"""

    LOAD_BALANCE_STRATEGY = "load_balance_strategy"
    HEALTH_CHECK_STRATEGY = "health_check_strategy"
    TASK_SCHEDULING_STRATEGY = "task_scheduling_strategy"
    METRICS_EXPORTER = "metrics_exporter"
    EXECUTOR = "executor"
    TOOL_INTEGRATION = "tool_integration"
    SPLIT_STRATEGY = "split_strategy"
    TEXT_PREPROCESSOR = "text_preprocessor"
    TEXT_POSTPROCESSOR = "text_postprocessor"
    METADATA_PROVIDER = "metadata_provider"
    AUDIO_CONVERTER = "audio_converter"
    TTS_BACKEND = "tts_backend"


class ExtensionPermission(Enum):
    """扩展权限枚举。

    定义扩展可以请求的权限类型，用于沙箱安全控制。
    """

    FILESYSTEM_READ = "filesystem_read"
    FILESYSTEM_WRITE = "filesystem_write"
    FILESYSTEM_DELETE = "filesystem_delete"
    NETWORK = "network"
    SUBPROCESS = "subprocess"
    CONFIG_READ = "config_read"
    CONFIG_WRITE = "config_write"
    EXEC = "exec"


class SecurityLevel(Enum):
    """安全级别枚举。

    定义沙箱对无权限操作的处理策略。
    """

    STRICT = "strict"
    WARN = "warn"
    ALLOW = "allow"
