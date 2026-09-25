"""验证 config.yml 注释保留 / Verify config.yml comment preservation.

对应 spec ``optimize-config-and-ext-system`` Task 1：
- ``save_config_to_file`` 使用 ``ruamel.yaml`` 往返保留注释
- ``config reset`` / ``config set`` / ``apply_preset`` 后注释保留
- ``config reset --factory`` 重写完整带注释模板
- ``ruamel`` 不可用时回退到 ``yaml.dump`` 并 WARNING
"""

from __future__ import annotations

import builtins
import logging
import sys
from pathlib import Path

import yaml as pyyaml

from src.config.manager_core import ConfigManager
from src.config.manager_io import save_commented_config, save_config_to_file
from src.config.presets import COMMENTED_YAML_TEMPLATE


# 一份带注释的最小配置 / A minimal commented config
_COMMENTED_CONFIG_YML = """\
# PPC_NEXT 配置文件 - 测试用 / Test config
# 顶部注释 / top comment
tts:
  # 语音模型 / voice model
  voice: zh-CN-XiaoxiaoNeural
  # 并发数 / concurrency
  concurrency: 8
  preset: balanced
version: 11.0.0rc1
"""


def _write_commented_config(path: Path) -> None:
    path.write_text(_COMMENTED_CONFIG_YML, encoding="utf-8")


def test_reset_preserves_comments(tmp_path: Path) -> None:
    """save_config_to_file 后 config.yml 注释应保留。"""
    cfg_path = tmp_path / "config.yml"
    _write_commented_config(cfg_path)

    mgr = ConfigManager(config_dir=tmp_path)
    # 触发一次保存（模拟 reset / apply_preset 后的保存）
    assert save_config_to_file(mgr) is True

    content = cfg_path.read_text(encoding="utf-8")
    # 注释行应保留
    assert "# 顶部注释 / top comment" in content
    assert "# 语音模型 / voice model" in content
    assert "# 并发数 / concurrency" in content
    # 值应正确
    parsed = pyyaml.safe_load(content)
    assert parsed["tts"]["voice"] == "zh-CN-XiaoxiaoNeural"
    assert parsed["tts"]["concurrency"] == 8


def test_set_single_field_preserves_comments(tmp_path: Path) -> None:
    """修改单字段后其它注释应保留。"""
    cfg_path = tmp_path / "config.yml"
    _write_commented_config(cfg_path)

    mgr = ConfigManager(config_dir=tmp_path)
    # 修改单字段（persist=True 会触发 save_config_to_file）
    assert mgr.set("tts.voice", "zh-CN-YunxiNeural", persist=True) is True

    content = cfg_path.read_text(encoding="utf-8")
    # 其它注释应保留
    assert "# 顶部注释 / top comment" in content
    assert "# 并发数 / concurrency" in content
    # 新值应写入
    parsed = pyyaml.safe_load(content)
    assert parsed["tts"]["voice"] == "zh-CN-YunxiNeural"
    # 未修改的字段应保持
    assert parsed["tts"]["concurrency"] == 8


def test_factory_reset_writes_full_template(tmp_path: Path) -> None:
    """config reset --factory 应重写完整带注释模板。

    ``handle_config(action='reset', factory=True)`` 的核心流程是
    ``save_commented_config(manager)`` + ``manager.reload()``，此处直接
    验证该机制。
    """
    cfg_path = tmp_path / "config.yml"
    # 先写入精简的带注释配置
    _write_commented_config(cfg_path)

    mgr = ConfigManager(config_dir=tmp_path)
    # 模拟 config reset --factory：重写完整带注释模板并重载
    save_commented_config(mgr)
    mgr.reload()

    content = cfg_path.read_text(encoding="utf-8")
    # 应等于完整模板
    assert content == COMMENTED_YAML_TEMPLATE
    # 应包含模板标志性注释
    assert "PPC_NEXT 配置文件" in content
    # 内存配置应与模板同步（balanced 预设默认语音）
    assert mgr.get_config().tts.voice == "zh-CN-XiaoxiaoNeural"


def test_ruamel_unavailable_fallback(tmp_path: Path, monkeypatch, caplog) -> None:
    """ruamel 不可用时应回退到 yaml.dump 并记录 WARNING。"""
    cfg_path = tmp_path / "config.yml"
    _write_commented_config(cfg_path)
    mgr = ConfigManager(config_dir=tmp_path)

    # 屏蔽 builtins.__import__ 中的 ruamel 导入
    real_import = builtins.__import__

    def _blocking_import(name: str, *args, **kwargs):
        if name == "ruamel" or name.startswith("ruamel."):
            raise ImportError("simulated ruamel unavailability")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _blocking_import)
    # 临时移除 ruamel 模块缓存，迫使 _save_with_ruamel 重新 import
    ruamel_cache = {k: sys.modules.pop(k) for k in list(sys.modules) if k.startswith("ruamel")}

    try:
        with caplog.at_level(logging.WARNING, logger="src.config.manager_io"):
            result = save_config_to_file(mgr)
    finally:
        # 恢复 ruamel 模块缓存（monkeypatch 会自动恢复 __import__）
        sys.modules.update(ruamel_cache)

    assert result is True
    # 回退时应记录 WARNING（包含 "ruamel"）
    assert any(
        "ruamel" in rec.message.lower() for rec in caplog.records
    ), f"expected ruamel fallback WARNING, got: {[r.message for r in caplog.records]}"

    # 文件应被写入（fallback 成功），但注释丢失（yaml.dump 不保留注释）
    content = cfg_path.read_text(encoding="utf-8")
    parsed = pyyaml.safe_load(content)
    assert parsed["tts"]["voice"] == "zh-CN-XiaoxiaoNeural"
    assert "# 顶部注释" not in content
