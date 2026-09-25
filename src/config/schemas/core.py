"""应用级配置段 Schema（core / split / batch / performance / network / features / extensions）"""

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.config.schemas.rules import CustomRule


class LogLevel(str, Enum):
    """日志级别"""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class CoreConfig(BaseModel):
    """核心配置"""

    model_config = ConfigDict(use_enum_values=False)

    log_level: LogLevel = Field(default=LogLevel.INFO, description="日志级别")
    progress_interval: int = Field(default=10, ge=1, le=100, description="进度回调触发频率(每N个)")


class SplitConfig(BaseModel):
    """分割配置"""

    preset: str = Field(default="chinese_novel", description="章节预设: chinese_novel | english_novel | default")
    min_chapter_length: int = Field(default=100, ge=10, description="最小章节长度")
    encoding_fallback: list[str] = Field(default_factory=lambda: ["utf-8", "gbk", "gb2312"], description="编码回退列表")
    encoding_detect_buffer: int = Field(default=1024, ge=256, le=8192, description="编码检测缓冲区大小(字节)")
    max_filename_length: int = Field(default=100, ge=50, le=200, description="文件名最大长度")
    custom_rules: list[CustomRule] = Field(default_factory=list, description="自定义分割规则列表")
    add_title_separator: bool = Field(default=True, description="是否在章节名后添加等于号分隔符")
    hierarchical_split: bool = Field(default=False, description="启用卷章体层级分割")
    volume_dir_prefix: str = Field(default="{volume}", description="卷目录名前缀格式")
    chapter_file_prefix: str = Field(
        default="{index:03d}", description="章节文件名序号格式（可含 {index} 占位符）"
    )


class BatchConfig(BaseModel):
    """批量处理配置"""

    max_size_mb: int = Field(default=95, ge=1, description="单批次最大大小(MB)")
    max_files_per_batch: int = Field(default=500, ge=1, description="每批次最大文件数")


class PerformanceConfig(BaseModel):
    """性能配置"""

    max_file_cache_size: int = Field(default=100, description="最大文件缓存数")
    stream_flush_threshold: int = Field(default=10, ge=1, description="进度流刷新阈值(每N个任务)")


class NetworkConfig(BaseModel):
    """网络配置"""

    timeout: int = Field(default=5, description="超时时间(秒)")


class ExtensionConfig(BaseModel):
    """用户自定义扩展配置"""

    enabled: bool = Field(default=True, description="启用扩展系统（false 时不加载任何扩展）")
    extension_dirs: list[str] = Field(
        default_factory=lambda: ["extensions"], description="扩展目录列表（相对路径或绝对路径）"
    )
    auto_load: bool = Field(default=True, description="CLI 启动时自动加载扩展")

    @model_validator(mode="before")
    @classmethod
    def _drop_legacy_installed_extensions(cls, data: Any) -> Any:
        """旧 config.yml 中的 installed_extensions 字段已废弃，静默忽略以保证兼容性。

        installed_extensions 原先记录在 config.yml 中，现已改为由扩展目录下的
        ``.installed`` 标记文件管理（见 ExtensionPackageManager）。
        """
        if isinstance(data, dict) and "installed_extensions" in data:
            data = dict(data)
            data.pop("installed_extensions", None)
        return data


class FanqieConfig(BaseModel):
    """番茄小说下载器（fanqie 扩展）配置

    覆盖 Web API 与 CLI 中原本硬编码的下载/启动参数默认值。
    """

    use_mirror: bool = Field(default=True, description="下载时优先使用镜像源 / Prefer mirror when downloading")
    mirror: str = Field(default="gh.llkk.cc", description="GitHub 下载镜像域名 / GitHub download mirror domain")
    prefer_musl: bool = Field(
        default=False, description="Linux amd64 优先选择 musl 静态构建 / Prefer musl build on Linux amd64"
    )
    server_host: str = Field(default="127.0.0.1", description="启动番茄下载器服务时的监听地址")
    server_port: int = Field(default=18423, ge=1, le=65535, description="启动番茄下载器服务时的监听端口")
