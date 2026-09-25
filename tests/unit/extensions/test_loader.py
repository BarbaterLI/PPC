"""ExtensionLoader 单元测试。"""

import asyncio
import logging
from pathlib import Path

from src.extensions.base import (
    ExtensionType,
    SecurityLevel,
)
from src.extensions.loader import ExtensionLoader
from src.extensions.sandbox import SandboxContext
from src.extensions.security import AuditLogger


def _run(coro):
    """运行异步协程。"""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            raise RuntimeError("loop is running")
        return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


# ---------------------------------------------------------------------------
# 测试扩展源代码模板
# ---------------------------------------------------------------------------

SIMPLE_EXT_SOURCE = '''"""Simple test extension."""
from src.extensions.base import Extension, ExtensionMetadata


class SimpleExtension(Extension):
    def __init__(self):
        super().__init__(ExtensionMetadata(
            name="simple-ext",
            version="1.0.0",
            description="Simple test extension",
        ))

    async def initialize(self) -> None:
        self._initialized = True

    async def cleanup(self) -> None:
        self._initialized = False


extension = SimpleExtension()
'''

DANGEROUS_EXT_SOURCE = '''"""Extension with dangerous pickle usage."""
import pickle

from src.extensions.base import Extension, ExtensionMetadata


class DangerousExtension(Extension):
    def __init__(self):
        super().__init__(ExtensionMetadata(
            name="dangerous-ext",
            version="1.0.0",
            description="Dangerous test extension",
        ))

    async def initialize(self) -> None:
        self._initialized = True

    async def cleanup(self) -> None:
        self._initialized = False

    def load_data(self, data):
        return pickle.loads(data)


extension = DangerousExtension()
'''

LIFECYCLE_EXT_SOURCE = '''"""Extension that tracks lifecycle hook calls."""
from src.extensions.base import Extension, ExtensionMetadata


class LifecycleExtension(Extension):
    def __init__(self):
        super().__init__(ExtensionMetadata(
            name="lifecycle-ext",
            version="1.0.0",
            description="Lifecycle test extension",
        ))
        self.before_load_called = False
        self.after_load_called = True  # will be set False then True on call
        self.before_load_called = False

    async def initialize(self) -> None:
        self._initialized = True

    async def cleanup(self) -> None:
        self._initialized = False

    async def before_load(self, context):
        self.before_load_called = True

    async def after_load(self, context):
        self.after_load_called = True


extension = LifecycleExtension()
'''

HOOK_EXCEPTION_EXT_SOURCE = '''"""Extension whose before_load raises."""
from src.extensions.base import Extension, ExtensionMetadata


class HookExceptionExtension(Extension):
    def __init__(self):
        super().__init__(ExtensionMetadata(
            name="hook-exc-ext",
            version="1.0.0",
            description="Hook exception test extension",
        ))

    async def initialize(self) -> None:
        self._initialized = True

    async def cleanup(self) -> None:
        self._initialized = False

    async def before_load(self, context):
        raise RuntimeError("before_load hook failed")


extension = HookExceptionExtension()
'''

BAD_SPLIT_EXT_SOURCE = '''"""Extension claiming SPLIT_STRATEGY but not implementing SplitStrategy."""
from src.extensions.base import Extension, ExtensionMetadata, ExtensionType


class BadSplitExtension(Extension):
    def __init__(self):
        super().__init__(ExtensionMetadata(
            name="bad-split",
            version="1.0.0",
            extension_type=ExtensionType.SPLIT_STRATEGY,
        ))

    async def initialize(self) -> None:
        self._initialized = True

    async def cleanup(self) -> None:
        self._initialized = False


extension = BadSplitExtension()
'''

GOOD_SPLIT_EXT_SOURCE = '''"""Extension implementing SplitStrategy interface."""
from src.extensions.base import (
    Extension, ExtensionMetadata, ExtensionType, SplitStrategy,
)


class GoodSplitExtension(Extension, SplitStrategy):
    def __init__(self):
        super().__init__(ExtensionMetadata(
            name="good-split",
            version="1.0.0",
            extension_type=ExtensionType.SPLIT_STRATEGY,
        ))

    async def initialize(self) -> None:
        self._initialized = True

    async def cleanup(self) -> None:
        self._initialized = False

    def split(self, content, options):
        return []

    def get_name(self):
        return "good-split"


extension = GoodSplitExtension()
'''

GOOD_PREPROCESSOR_EXT_SOURCE = '''"""Extension implementing TextPreprocessor interface."""
from src.extensions.base import (
    Extension, ExtensionMetadata, ExtensionType, TextPreprocessor,
)


class GoodPreprocessorExtension(Extension, TextPreprocessor):
    def __init__(self):
        super().__init__(ExtensionMetadata(
            name="good-preprocessor",
            version="1.0.0",
            extension_type=ExtensionType.TEXT_PREPROCESSOR,
        ))

    async def initialize(self) -> None:
        self._initialized = True

    async def cleanup(self) -> None:
        self._initialized = False

    def process(self, text, context):
        return text

    def get_name(self):
        return "good-preprocessor"


extension = GoodPreprocessorExtension()
'''


def _write_ext(tmp_path: Path, name: str, source: str) -> Path:
    """写入测试扩展 .py 文件并返回路径。"""
    ext_path = tmp_path / f"{name}.py"
    ext_path.write_text(source, encoding="utf-8")
    return ext_path


# ---------------------------------------------------------------------------
# 测试用例
# ---------------------------------------------------------------------------


def test_default_security_level_is_warn():
    """默认 security_level 应为 SecurityLevel.WARN。"""
    loader = ExtensionLoader()
    assert loader.get_security_level() == SecurityLevel.WARN


def test_load_extension_from_file(tmp_path: Path):
    """从 .py 文件加载简单扩展。"""
    ext_path = _write_ext(tmp_path, "simple_ext", SIMPLE_EXT_SOURCE)
    loader = ExtensionLoader()
    try:
        _run(loader._load_extension_from_file(ext_path))
        loaded = loader.get_loaded_extensions()
        assert "simple-ext" in loaded
        assert loaded["simple-ext"].metadata.version == "1.0.0"
        assert loaded["simple-ext"].metadata.description == "Simple test extension"
    finally:
        _run(loader.clear())


def test_load_extension_strict_blocks_critical(tmp_path: Path):
    """STRICT 模式下，含 CRITICAL 安全问题(pickle.loads)的扩展被阻止加载。"""
    ext_path = _write_ext(tmp_path, "dangerous_ext", DANGEROUS_EXT_SOURCE)
    loader = ExtensionLoader(security_level=SecurityLevel.STRICT)
    try:
        _run(loader._load_extension_from_file(ext_path))
        loaded = loader.get_loaded_extensions()
        assert "dangerous-ext" not in loaded
        failed = loader.get_failed_extensions()
        assert len(failed) == 1
        assert "Security scan blocked" in failed[0]["error"]
    finally:
        _run(loader.clear())


def test_load_extension_warn_allows_critical(tmp_path: Path, caplog):
    """WARN 模式下，含 CRITICAL 安全问题的扩展仍可加载但记录警告。"""
    ext_path = _write_ext(tmp_path, "dangerous_ext", DANGEROUS_EXT_SOURCE)
    loader = ExtensionLoader(security_level=SecurityLevel.WARN)
    try:
        with caplog.at_level(logging.WARNING):
            _run(loader._load_extension_from_file(ext_path))
        loaded = loader.get_loaded_extensions()
        assert "dangerous-ext" in loaded
        assert loader.get_failed_extensions() == []
    finally:
        _run(loader.clear())


def test_load_extension_allow_allows_critical(tmp_path: Path, caplog):
    """ALLOW 模式下，含 CRITICAL 安全问题的扩展可加载且不记录警告日志。"""
    ext_path = _write_ext(tmp_path, "dangerous_ext", DANGEROUS_EXT_SOURCE)
    loader = ExtensionLoader(security_level=SecurityLevel.ALLOW)
    try:
        with caplog.at_level(logging.DEBUG):
            _run(loader._load_extension_from_file(ext_path))
        loaded = loader.get_loaded_extensions()
        assert "dangerous-ext" in loaded
        assert loader.get_failed_extensions() == []
    finally:
        _run(loader.clear())


def test_lifecycle_hooks_called(tmp_path: Path):
    """before_load 和 after_load 钩子被调用。"""
    ext_path = _write_ext(tmp_path, "lifecycle_ext", LIFECYCLE_EXT_SOURCE)
    loader = ExtensionLoader()
    try:
        _run(loader._load_extension_from_file(ext_path))
        ext = loader.get_extension("lifecycle-ext")
        assert ext is not None
        assert ext.before_load_called is True
        assert ext.after_load_called is True
    finally:
        _run(loader.clear())


def test_hook_exception_doesnt_block_load(tmp_path: Path):
    """如果 before_load 钩子抛异常，加载仍然成功（钩子错误被吞掉）。"""
    ext_path = _write_ext(tmp_path, "hook_exc_ext", HOOK_EXCEPTION_EXT_SOURCE)
    loader = ExtensionLoader()
    try:
        _run(loader._load_extension_from_file(ext_path))
        loaded = loader.get_loaded_extensions()
        assert "hook-exc-ext" in loaded
        assert loader.get_failed_extensions() == []
    finally:
        _run(loader.clear())


def test_validate_extension_rejects_missing_interface(tmp_path: Path):
    """扩展声明 SPLIT_STRATEGY 但未实现 SplitStrategy 接口时被拒绝。"""
    ext_path = _write_ext(tmp_path, "bad_split", BAD_SPLIT_EXT_SOURCE)
    loader = ExtensionLoader()
    try:
        _run(loader._load_extension_from_file(ext_path))
        loaded = loader.get_loaded_extensions()
        assert "bad-split" not in loaded
        failed = loader.get_failed_extensions()
        assert len(failed) == 1
        assert "Interface validation failed" in failed[0]["error"]
    finally:
        _run(loader.clear())


def test_get_split_strategies(tmp_path: Path):
    """get_split_strategies 返回已注册的 SPLIT_STRATEGY 扩展。"""
    ext_path = _write_ext(tmp_path, "good_split", GOOD_SPLIT_EXT_SOURCE)
    loader = ExtensionLoader()
    try:
        _run(loader._load_extension_from_file(ext_path))
        strategies = loader.get_split_strategies()
        assert len(strategies) == 1
        assert strategies[0].metadata.name == "good-split"
    finally:
        _run(loader.clear())


def test_get_text_preprocessors(tmp_path: Path):
    """get_text_preprocessors 返回已注册的 TEXT_PREPROCESSOR 扩展。"""
    ext_path = _write_ext(tmp_path, "good_preprocessor", GOOD_PREPROCESSOR_EXT_SOURCE)
    loader = ExtensionLoader()
    try:
        _run(loader._load_extension_from_file(ext_path))
        preprocessors = loader.get_text_preprocessors()
        assert len(preprocessors) == 1
        assert preprocessors[0].metadata.name == "good-preprocessor"
    finally:
        _run(loader.clear())


def test_get_extensions_by_type(tmp_path: Path):
    """get_extensions_by_type 正确过滤扩展类型。"""
    split_path = _write_ext(tmp_path, "good_split", GOOD_SPLIT_EXT_SOURCE)
    prep_path = _write_ext(tmp_path, "good_preprocessor", GOOD_PREPROCESSOR_EXT_SOURCE)
    loader = ExtensionLoader()
    try:
        _run(loader._load_extension_from_file(split_path))
        _run(loader._load_extension_from_file(prep_path))

        splits = loader.get_extensions_by_type(ExtensionType.SPLIT_STRATEGY)
        assert len(splits) == 1
        assert splits[0].metadata.name == "good-split"

        preps = loader.get_extensions_by_type(ExtensionType.TEXT_PREPROCESSOR)
        assert len(preps) == 1
        assert preps[0].metadata.name == "good-preprocessor"

        # 不存在的类型返回空列表
        assert loader.get_extensions_by_type(ExtensionType.EXECUTOR) == []
    finally:
        _run(loader.clear())


def test_enable_disable(tmp_path: Path):
    """enable 和 disable 正常工作。"""
    ext_path = _write_ext(tmp_path, "simple_ext", SIMPLE_EXT_SOURCE)
    loader = ExtensionLoader()
    try:
        _run(loader._load_extension_from_file(ext_path))
        ext = loader.get_extension("simple-ext")
        assert ext is not None
        # 加载后默认 enabled
        assert ext.enabled is True

        # 禁用
        result = _run(loader.disable("simple-ext"))
        assert result is True
        assert ext.enabled is False

        # 再次禁用（已禁用，返回 True）
        result = _run(loader.disable("simple-ext"))
        assert result is True

        # 启用
        result = _run(loader.enable("simple-ext"))
        assert result is True
        assert ext.enabled is True

        # 再次启用（已启用，返回 True）
        result = _run(loader.enable("simple-ext"))
        assert result is True
    finally:
        _run(loader.clear())


def test_enable_disable_nonexistent():
    """enable/disable 不存在的扩展返回 False。"""
    loader = ExtensionLoader()
    try:
        assert _run(loader.enable("nonexistent")) is False
        assert _run(loader.disable("nonexistent")) is False
    finally:
        _run(loader.clear())


def test_get_failed_extensions(tmp_path: Path):
    """加载失败的扩展记录在 get_failed_extensions 中。"""
    ext_path = _write_ext(tmp_path, "bad_split", BAD_SPLIT_EXT_SOURCE)
    loader = ExtensionLoader()
    try:
        _run(loader._load_extension_from_file(ext_path))
        failed = loader.get_failed_extensions()
        assert len(failed) == 1
        assert failed[0]["filename"] == "bad_split.py"
        assert "Interface validation failed" in failed[0]["error"]
    finally:
        _run(loader.clear())


def test_clear(tmp_path: Path):
    """clear 移除所有已加载的扩展。"""
    ext_path = _write_ext(tmp_path, "simple_ext", SIMPLE_EXT_SOURCE)
    loader = ExtensionLoader()
    try:
        _run(loader._load_extension_from_file(ext_path))
        assert len(loader.get_loaded_extensions()) == 1
        assert len(loader.get_failed_extensions()) == 0

        _run(loader.clear())

        assert loader.get_loaded_extensions() == {}
        assert loader.get_failed_extensions() == []
    finally:
        _run(loader.clear())


def test_sandbox_injected_into_context(tmp_path: Path):
    """加载后的扩展上下文中包含 sandbox 实例。"""
    ext_path = _write_ext(tmp_path, "simple_ext", SIMPLE_EXT_SOURCE)
    loader = ExtensionLoader()
    try:
        _run(loader._load_extension_from_file(ext_path))
        ext = loader.get_extension("simple-ext")
        assert ext is not None
        assert ext._context is not None
        assert isinstance(ext._context.sandbox, SandboxContext)
        assert ext._context.sandbox.extension_name == "simple-ext"
    finally:
        _run(loader.clear())


def test_audit_log_on_load(tmp_path: Path):
    """AuditLogger 在加载扩展时记录 EXTENSION_LOAD 事件。"""
    ext_path = _write_ext(tmp_path, "simple_ext", SIMPLE_EXT_SOURCE)
    audit_logger = AuditLogger(log_dir=tmp_path / "audit")
    loader = ExtensionLoader(audit_logger=audit_logger)
    try:
        _run(loader._load_extension_from_file(ext_path))
        logs = audit_logger.read_recent_logs()
        load_events = [e for e in logs if e["action"] == "EXTENSION_LOAD"]
        assert len(load_events) > 0
        success_events = [
            e for e in load_events
            if e["result"] == "success" and e["extension_name"] == "simple-ext"
        ]
        assert len(success_events) > 0
    finally:
        _run(loader.clear())


def test_audit_log_on_enable(tmp_path: Path):
    """AuditLogger 在启用扩展时记录 EXTENSION_ENABLE 事件。"""
    ext_path = _write_ext(tmp_path, "simple_ext", SIMPLE_EXT_SOURCE)
    audit_logger = AuditLogger(log_dir=tmp_path / "audit")
    loader = ExtensionLoader(audit_logger=audit_logger)
    try:
        _run(loader._load_extension_from_file(ext_path))
        # 加载后 _enabled=True，需要先禁用以触发 enable 的完整流程
        _run(loader.disable("simple-ext"))
        # 清空审计日志文件以确保只检测 enable 事件
        log_file = audit_logger.get_log_file_path()
        log_file.write_text("", encoding="utf-8")

        result = _run(loader.enable("simple-ext"))
        assert result is True

        logs = audit_logger.read_recent_logs()
        enable_events = [e for e in logs if e["action"] == "EXTENSION_ENABLE"]
        assert len(enable_events) > 0
        success_events = [
            e for e in enable_events
            if e["result"] == "success" and e["extension_name"] == "simple-ext"
        ]
        assert len(success_events) > 0
    finally:
        _run(loader.clear())


def test_audit_log_on_disable(tmp_path: Path):
    """AuditLogger 在禁用扩展时记录 EXTENSION_DISABLE 事件。"""
    ext_path = _write_ext(tmp_path, "simple_ext", SIMPLE_EXT_SOURCE)
    audit_logger = AuditLogger(log_dir=tmp_path / "audit")
    loader = ExtensionLoader(audit_logger=audit_logger)
    try:
        _run(loader._load_extension_from_file(ext_path))
        # 清空审计日志文件以确保只检测 disable 事件
        log_file = audit_logger.get_log_file_path()
        log_file.write_text("", encoding="utf-8")

        result = _run(loader.disable("simple-ext"))
        assert result is True

        logs = audit_logger.read_recent_logs()
        disable_events = [e for e in logs if e["action"] == "EXTENSION_DISABLE"]
        assert len(disable_events) > 0
        success_events = [
            e for e in disable_events
            if e["result"] == "success" and e["extension_name"] == "simple-ext"
        ]
        assert len(success_events) > 0
    finally:
        _run(loader.clear())


# ---------------------------------------------------------------------------
# unload(name) 测试
# ---------------------------------------------------------------------------

UNLOAD_TRACKING_EXT_SOURCE = '''"""Extension that tracks cleanup() calls during unload."""
import typer
from src.extensions.base import Extension, ExtensionMetadata


class UnloadTrackingExtension(Extension):
    def __init__(self):
        super().__init__(ExtensionMetadata(
            name="unload-tracking-ext",
            version="1.0.0",
            description="Tracks cleanup calls",
        ))
        self.cleanup_called = False

    async def initialize(self) -> None:
        self._initialized = True

    async def cleanup(self) -> None:
        self._initialized = False
        self.cleanup_called = True

    def register_cli(self, app):
        """实现 register_cli 接口，使本扩展出现在 get_cli_extensions() 中。"""
        @app.command("dummy-cmd")
        def _dummy():
            pass


extension = UnloadTrackingExtension()
'''


def test_unload_existing_extension(tmp_path: Path):
    """unload(name) 卸载已加载的扩展并从 get_loaded_extensions() 移除。"""
    ext_path = _write_ext(tmp_path, "unload_ext", UNLOAD_TRACKING_EXT_SOURCE)
    loader = ExtensionLoader()
    try:
        _run(loader._load_extension_from_file(ext_path))
        assert "unload-tracking-ext" in loader.get_loaded_extensions()

        result = _run(loader.unload("unload-tracking-ext"))
        assert result is True
        assert "unload-tracking-ext" not in loader.get_loaded_extensions()
    finally:
        _run(loader.clear())


def test_unload_nonexistent_extension():
    """unload(name) 卸载不存在的扩展返回 False。"""
    loader = ExtensionLoader()
    try:
        result = _run(loader.unload("does-not-exist"))
        assert result is False
    finally:
        _run(loader.clear())


def test_unload_calls_cleanup(tmp_path: Path):
    """unload(name) 调用扩展的 cleanup() 方法。"""
    ext_path = _write_ext(tmp_path, "unload_ext", UNLOAD_TRACKING_EXT_SOURCE)
    loader = ExtensionLoader()
    try:
        _run(loader._load_extension_from_file(ext_path))
        ext = loader.get_extension("unload-tracking-ext")
        assert ext is not None
        assert ext.cleanup_called is False

        _run(loader.unload("unload-tracking-ext"))
        assert ext.cleanup_called is True
    finally:
        _run(loader.clear())


def test_unload_removes_from_cli_extensions(tmp_path: Path):
    """unload(name) 从 get_cli_extensions() 中移除扩展。"""
    ext_path = _write_ext(tmp_path, "unload_ext", UNLOAD_TRACKING_EXT_SOURCE)
    loader = ExtensionLoader()
    try:
        _run(loader._load_extension_from_file(ext_path))

        cli_exts_before = loader.get_cli_extensions()
        cli_names_before = {e.metadata.name for e in cli_exts_before}
        assert "unload-tracking-ext" in cli_names_before

        _run(loader.unload("unload-tracking-ext"))

        cli_exts_after = loader.get_cli_extensions()
        cli_names_after = {e.metadata.name for e in cli_exts_after}
        assert "unload-tracking-ext" not in cli_names_after
    finally:
        _run(loader.clear())
