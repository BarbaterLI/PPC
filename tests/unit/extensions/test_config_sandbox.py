"""Extension.load_config / save_config 沙箱拦截测试。

验证 :meth:`Extension.load_config` 与 :meth:`Extension.save_config` 在不同
安全级别（STRICT / WARN / ALLOW）与不同权限组合下的实际行为：

* ``load_config`` 优先检查文件是否存在（不存在直接返回 ``{}``，不触发权限检查）；
  存在时通过 ``sandbox.safe_read_file`` 走权限校验。
* ``save_config`` 通过 ``sandbox.safe_write_file`` 走权限校验，所有异常
  被捕获并记录 warning，不向调用方抛出。

测试构造一个最小 Extension 子类，手动注入 :class:`SandboxContext`，
并把 ``get_extension_data_dir`` 重定向到 ``tmp_path``，避免污染全局
``extensions_data`` 目录。
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
import yaml

from src.extensions.base import (
    Extension,
    ExtensionContext,
    ExtensionMetadata,
    ExtensionPermission,
    SecurityLevel,
)
from src.extensions.sandbox import SandboxConfig, SandboxContext
from src.extensions.security import AuditLogger

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def audit_logger(tmp_path: Path) -> AuditLogger:
    """Fresh AuditLogger writing to an isolated tmp directory."""
    return AuditLogger(log_dir=tmp_path / "audit")


# ---------------------------------------------------------------------------
# 测试用 Extension 子类
# ---------------------------------------------------------------------------


class _StubExtension(Extension):
    """最小 Extension 子类，仅用于测试 load_config/save_config。"""

    def __init__(self, name: str = "config-test-ext") -> None:
        super().__init__(ExtensionMetadata(name=name, version="1.0.0"))

    async def initialize(self) -> None:
        self._initialized = True

    async def cleanup(self) -> None:
        self._initialized = False


# ---------------------------------------------------------------------------
# 辅助构造
# ---------------------------------------------------------------------------


def _make_extension(
    data_dir: Path,
    audit_logger: AuditLogger,
    permissions: list[ExtensionPermission],
    *,
    security_level: SecurityLevel = SecurityLevel.STRICT,
    name: str = "config-test-ext",
) -> _StubExtension:
    """构造一个带真实 SandboxContext 的 Extension 实例。

    ``get_extension_data_dir`` 被重定向到 ``data_dir``，避免写入全局
    ``extensions_data`` 目录。
    """
    ext = _StubExtension(name=name)
    ext.get_extension_data_dir = lambda: data_dir  # type: ignore[method-assign]
    sandbox = SandboxContext(SandboxConfig(
        extension_name=name,
        permissions=permissions,
        audit_logger=audit_logger,
        security_level=security_level,
        data_dir=data_dir,
    ))
    ext._context = ExtensionContext(
        loader=None,  # type: ignore[arg-type]
        extension_dir=data_dir,
        data_dir=data_dir,
        sandbox=sandbox,
    )
    return ext


# ---------------------------------------------------------------------------
# load_config
# ---------------------------------------------------------------------------


def test_load_config_passes_with_filesystem_read_permission(
    tmp_path: Path, audit_logger: AuditLogger
) -> None:
    """有 FILESYSTEM_READ 权限时 load_config 返回保存的配置字典。"""
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    ext = _make_extension(
        data_dir, audit_logger, [ExtensionPermission.FILESYSTEM_READ]
    )

    config_file = data_dir / "config.yml"
    config_file.write_text(
        yaml.safe_dump({"key": "value", "nested": {"a": 1}}, allow_unicode=True),
        encoding="utf-8",
    )

    result = ext.load_config()
    assert result == {"key": "value", "nested": {"a": 1}}


def test_load_config_returns_empty_when_file_missing(
    tmp_path: Path, audit_logger: AuditLogger
) -> None:
    """配置文件缺失时 load_config 返回空字典（不崩溃、不触发权限检查）。"""
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    # 即使没有任何权限，文件不存在也应直接返回 {}
    ext = _make_extension(data_dir, audit_logger, [])

    result = ext.load_config()
    assert result == {}


# ---------------------------------------------------------------------------
# save_config
# ---------------------------------------------------------------------------


def test_save_config_blocked_in_strict_mode(
    tmp_path: Path,
    audit_logger: AuditLogger,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """STRICT 模式下，无 FILESYSTEM_WRITE 权限时 save_config 不写入文件。

    ``safe_write_file`` 抛出 PermissionError，``save_config`` 捕获后记录
    warning；配置文件不应被创建。
    """
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    ext = _make_extension(
        data_dir, audit_logger, [], security_level=SecurityLevel.STRICT
    )

    with caplog.at_level(logging.WARNING):
        ext.save_config({"key": "value"})

    config_file = data_dir / "config.yml"
    assert not config_file.exists(), "STRICT 模式下未授权写入不应创建文件"

    # save_config 捕获异常后记录 warning
    assert any(
        "failed to save config" in rec.message for rec in caplog.records
    ), "STRICT 模式下应记录保存失败警告"


def test_save_config_warned_in_warn_mode(
    tmp_path: Path,
    audit_logger: AuditLogger,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """WARN 模式下，无 FILESYSTEM_WRITE 权限时 save_config 仍写入文件。

    ``_check_permission`` 在 WARN 模式下返回 True（仅记录 warning），
    因此 ``safe_write_file`` 继续执行写入；配置文件应当被创建。
    """
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    ext = _make_extension(
        data_dir, audit_logger, [], security_level=SecurityLevel.WARN
    )

    with caplog.at_level(logging.WARNING):
        ext.save_config({"key": "value"})

    config_file = data_dir / "config.yml"
    assert config_file.exists(), "WARN 模式应允许写入（仅记录 warning）"
    saved = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    assert saved == {"key": "value"}

    # WARN 模式下应记录 missing permission 警告
    assert any(
        "missing permission" in rec.message.lower()
        for rec in caplog.records
    ), "WARN 模式应记录缺少权限的 warning"


def test_save_config_allowed_in_allow_mode(
    tmp_path: Path,
    audit_logger: AuditLogger,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """ALLOW 模式下，无 FILESYSTEM_WRITE 权限时 save_config 仍写入文件并记录审计。

    ``_check_permission`` 在 ALLOW 模式下返回 True，并调用
    ``log_permission_denied`` 记录审计事件；配置文件应当被创建。
    """
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    ext = _make_extension(
        data_dir, audit_logger, [], security_level=SecurityLevel.ALLOW
    )

    with caplog.at_level(logging.WARNING):
        ext.save_config({"key": "value"})

    config_file = data_dir / "config.yml"
    assert config_file.exists(), "ALLOW 模式应允许写入"
    saved = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    assert saved == {"key": "value"}

    # ALLOW 模式应在审计日志中记录 permission_denied 事件
    logs = audit_logger.read_recent_logs()
    denied_events = [
        e for e in logs
        if e.get("action") == "permission_denied"
        and e.get("result") == "denied"
    ]
    assert len(denied_events) > 0, (
        "ALLOW 模式应记录 permission_denied 审计事件，"
        f"实际日志条目: {logs}"
    )
