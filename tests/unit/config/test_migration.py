"""Unit tests for :mod:`src.config.migration`.

覆盖 PPC5/8 配置向 PPC_NEXT 迁移的能力。
"""

from __future__ import annotations

import configparser
from pathlib import Path

from src.config.migration import (
    VERSION_MAPPING,
    ConfigMigrator,
    migrate_ppc5_config,
)

# ---------------------------------------------------------------------------
# v5 → v6 迁移
# ---------------------------------------------------------------------------


class TestV5ToV6:
    def test_migrate_v5_ini(self, tmp_path: Path) -> None:
        v5 = tmp_path / "ppc5.ini"
        cp = configparser.ConfigParser()
        cp["tts"] = {
            "voice": "zh-CN-YunxiNeural",
            "concurrency": "6",
            "retries": "3",
            "timeout": "60",
            "max_segment_length": "2000",
        }
        cp["split"] = {
            "encoding_fallback": "utf-8, gbk",
            "min_chapter_length": "120",
        }
        with v5.open("w", encoding="utf-8") as f:
            cp.write(f)

        m = ConfigMigrator()
        result = m.migrate(v5, source_version="5.0")
        assert result  # 非空
        # 转换后应包含 tts / split 字段
        assert "tts" in result or "core" in result
        # v6 格式应包含 version
        assert "version" in result

    def test_migrate_v5_missing_file(self, tmp_path: Path) -> None:
        m = ConfigMigrator()
        result = m.migrate(tmp_path / "missing.ini", source_version="5.0")
        # 缺失文件不应抛异常，应返回空 dict
        assert result == {}

    def test_migrate_ppc5_function(self, tmp_path: Path) -> None:
        v5 = tmp_path / "ppc5.ini"
        cp = configparser.ConfigParser()
        cp["tts"] = {"voice": "x", "concurrency": "1", "retries": "1", "timeout": "30", "max_segment_length": "100"}
        with v5.open("w", encoding="utf-8") as f:
            cp.write(f)
        result = migrate_ppc5_config(v5)
        assert result


# ---------------------------------------------------------------------------
# v8 → v10 迁移
# ---------------------------------------------------------------------------


class TestV8ToV9:
    def test_migrate_v8_yaml(self, tmp_path: Path) -> None:
        v8 = tmp_path / "ppc8.yml"
        v8.write_text(
            "version: 8.1.0\ntts:\n  voice: zh-CN-XiaoxiaoNeural\n",
            encoding="utf-8",
        )
        m = ConfigMigrator()
        result = m.migrate(v8, source_version="8.0")
        assert result
        assert result.get("version") == "9.0.0"
        assert "_migrated_from" in result

    def test_migrate_v8_missing_file(self, tmp_path: Path) -> None:
        m = ConfigMigrator()
        result = m.migrate(tmp_path / "missing.yml", source_version="8.0")
        assert result == {}

    def test_migrate_v8_invalid_yaml(self, tmp_path: Path) -> None:
        v8 = tmp_path / "bad.yml"
        v8.write_text("this: is: not: valid: yaml: : :", encoding="utf-8")
        m = ConfigMigrator()
        result = m.migrate(v8, source_version="8.0")
        # 解析失败应返回空 dict，不抛异常
        assert result == {}


# ---------------------------------------------------------------------------
# 未知版本
# ---------------------------------------------------------------------------


class TestUnknownVersion:
    def test_unknown_version_returns_empty(self, tmp_path: Path) -> None:
        v = tmp_path / "x.ini"
        v.write_text("[tts]\nvoice = x\n", encoding="utf-8")
        m = ConfigMigrator()
        result = m.migrate(v, source_version="3.0")
        assert result == {}

    def test_version_mapping_has_expected_keys(self) -> None:
        # 验证映射表覆盖主流版本
        assert "5.0" in VERSION_MAPPING
        assert "8.0" in VERSION_MAPPING
        assert "11.0" in VERSION_MAPPING  # 11 -> 12.x


# ---------------------------------------------------------------------------
# v11 → v12 迁移
# ---------------------------------------------------------------------------


class TestV11ToV12:
    def test_migrate_v11_bumps_version_and_drops_dead_sections(self, tmp_path: Path) -> None:
        from src.version import __version__

        v11 = tmp_path / "config.yml"
        v11.write_text(
            "version: 11.0.0\n"
            "tts:\n  voice: zh-CN-YunxiNeural\n  concurrency: 6\n"
            "features: {a: 1}\n"
            "webhook: {url: x}\n"
            "rate_limit: {r: 5}\n",
            encoding="utf-8",
        )
        m = ConfigMigrator()
        result = m.migrate(v11, source_version="11.0")
        assert result
        assert result["version"] == __version__
        # 12.x 移除的死段被显式剥离
        assert "features" not in result
        assert "webhook" not in result
        assert "rate_limit" not in result
        # 既有配置字段保留
        assert result["tts"]["voice"] == "zh-CN-YunxiNeural"
        assert result["tts"]["concurrency"] == 6
        # 迁移元数据
        fm = result["_migrated_from"]
        assert fm["source_version"] == "11.0.0"
        assert fm["source"] == str(v11)

    def test_migrate_v11_missing_file(self, tmp_path: Path) -> None:
        m = ConfigMigrator()
        assert m.migrate(tmp_path / "missing.yml", source_version="11.0") == {}

    def test_migrate_v11_invalid_yaml(self, tmp_path: Path) -> None:
        v11 = tmp_path / "bad.yml"
        v11.write_text("a: b: c: :", encoding="utf-8")
        m = ConfigMigrator()
        assert m.migrate(v11, source_version="11.0") == {}

    def test_migrate_v11_non_mapping_top_level(self, tmp_path: Path) -> None:
        v11 = tmp_path / "list.yml"
        v11.write_text("- 1\n- 2\n", encoding="utf-8")
        m = ConfigMigrator()
        assert m.migrate(v11, source_version="11.0") == {}

    def test_config_manager_load_runs_v11_migration_chain(self, tmp_path: Path) -> None:
        """加载 v11 config.yml 时必须真正执行 v11→v12 迁移。

        触发条件：``config.yml`` 的 ``version: 11.0.0``（12.x 之前的 YAML）。
        后果：旧实现只从 PPC5 路径调用 ``migrate(..., "5.0")``，
        ``migrate_v11_to_v12`` 是死代码——死配置段不剥离、version 不升级、
        无 ``_migrated_from``。本测试从 ``ConfigManager`` 加载路径覆盖该链路。
        """
        from src.config.manager_core import ConfigManager
        from src.version import __version__

        (tmp_path / "config.yml").write_text(
            "version: 11.0.0\n"
            "tts:\n  voice: zh-CN-YunxiNeural\n  concurrency: 6\n"
            "features: {a: 1}\n"
            "webhook: {url: x}\n"
            "rate_limit: {r: 5}\n",
            encoding="utf-8",
        )

        mgr = ConfigManager(config_dir=tmp_path)
        cfg_dict = mgr._config_dict

        assert cfg_dict["version"] == __version__
        assert "features" not in cfg_dict
        assert "webhook" not in cfg_dict
        assert "rate_limit" not in cfg_dict
        assert cfg_dict["tts"]["voice"] == "zh-CN-YunxiNeural"
        assert cfg_dict["_migrated_from"]["source_version"] == "11.0.0"
        # Pydantic 模型也要能正常构造
        assert mgr.get_config().tts.concurrency == 6
