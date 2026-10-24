"""配置保存不得丢弃 schema 外未知键的测试。"""

from __future__ import annotations

from pathlib import Path

from src.config import ConfigManager


def test_save_preserves_unknown_keys(tmp_path: Path):
    """保存 config.yml 不得丢弃 schema 外的未知（legacy）键。

    Bug: ``manager_access._normalize_dict`` 显式保留未知顶层键（docstring
    明言“避免丢配置”），但 ``manager_io.save_config_to_file`` 却用
    ``manager._config.model_dump()``（只含 schema 字段）构造待写 dict，
    ``_update_commented_map`` 随即把 config_dict 里没有的键从文件删除 →
    内存保留、保存即丢，legacy 键（如 ``ui:``）在第一次 ``set`` 触发保存后
    从 config.yml 消失，与 _normalize_dict 的意图矛盾。
    修复: save 时把 ``_config_dict`` 中 schema 外的未知键合并进待写 dict。
    """
    d = tmp_path
    (d / "config.yml").write_text(
        "core:\n  log_level: info\nui:\n  theme: dark\n",
        encoding="utf-8",
    )

    mgr = ConfigManager(d)
    assert "ui" in mgr._config_dict, "加载后内存应保留未知键"

    assert mgr.set("tts.concurrency", 4) is True  # 触发保存

    saved = (d / "config.yml").read_text(encoding="utf-8")
    assert "theme" in saved and "ui" in saved, f"未知键 ui 在保存时被丢弃:\n{saved[:200]}"


def test_save_drops_deprecated_shared_connector(tmp_path: Path):
    """已废弃的 tts.anti_risk_control.shared_connector 不得再写回 config.yml。

    该开关从未真正生效（逐流 ClientSession 会关闭共享 connector），继续写出
    会让用户误以为连接复用已开启。旧配置文件中的该键在首次保存时应被剔除。
    """
    d = tmp_path
    (d / "config.yml").write_text(
        "tts:\n  anti_risk_control:\n    shared_connector: true\n",
        encoding="utf-8",
    )

    mgr = ConfigManager(d)
    assert mgr.set("tts.concurrency", 4) is True  # 触发保存

    saved = (d / "config.yml").read_text(encoding="utf-8")
    assert "shared_connector" not in saved, (
        f"已废弃的 shared_connector 仍被写出:\n{saved[:400]}"
    )
