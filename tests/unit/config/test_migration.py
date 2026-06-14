"""Unit tests for :mod:`src_m.config.migration`.

覆盖 PPC5/8 配置向 PPC10 迁移的能力。
"""
from __future__ import annotations

import configparser
from pathlib import Path

import pytest

from src_m.config.migration import (
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
