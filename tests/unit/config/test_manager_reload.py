"""验证 ConfigManager.reload() 清理 _temp_overrides 与缓存一致性 / Verify reload clears overrides + cache.

对应 spec ``finalize-followup-updates`` Task 2：修复 ``reload()`` 在 ``_load()``
之后才清理 ``_temp_overrides``、且未失效 ``_config_dict_hash`` 的缓存不一致 bug。
"""

from __future__ import annotations

from pathlib import Path

from src.config.manager_core import ConfigManager


_CONFIG_YML = """\
tts:
  voice: zh-CN-XiaoxiaoNeural
  concurrency: 4
"""


def _write_config(path: Path) -> None:
    path.write_text(_CONFIG_YML, encoding="utf-8")


class TestReloadCacheConsistency:
    def test_reload_clears_temp_overrides(self, tmp_path: Path) -> None:
        """reload() 后 _temp_overrides 应为空。"""
        cfg_path = tmp_path / "config.yml"
        _write_config(cfg_path)
        mgr = ConfigManager(config_dir=tmp_path)
        mgr.set_temp("tts.voice", "zh-CN-YunxiNeural")
        assert len(mgr._temp_overrides) == 1
        mgr.reload()
        assert len(mgr._temp_overrides) == 0

    def test_reload_returns_clean_config(self, tmp_path: Path) -> None:
        """reload() 后 get_config() 应返回未应用 override 的纯净配置。"""
        cfg_path = tmp_path / "config.yml"
        _write_config(cfg_path)
        mgr = ConfigManager(config_dir=tmp_path)
        original_voice = mgr.get_config().tts.voice
        mgr.set_temp("tts.voice", "zh-CN-YunxiNeural")
        # 缓存可能命中 — 强制 reload
        mgr.reload()
        after = mgr.get_config()
        assert after.tts.voice == original_voice
        assert after.tts.voice != "zh-CN-YunxiNeural"

    def test_reload_invalidates_hash_cache(self, tmp_path: Path) -> None:
        """reload() 后 _config_dict_hash 应为 None 或重建后的新哈希（非旧 override 哈希）。"""
        cfg_path = tmp_path / "config.yml"
        _write_config(cfg_path)
        mgr = ConfigManager(config_dir=tmp_path)
        mgr.set_temp("tts.voice", "zh-CN-YunxiNeural")
        # set_temp 已将 _config_dict_hash 置 None；显式 _load() 计算带 override 的哈希
        # （get_config 仅在 TTL 过期时才触发 _load，故此处直接调用以稳定测试）
        mgr._load()
        hash_with_override = mgr._config_dict_hash
        assert hash_with_override is not None
        mgr.reload()
        # reload 后立即检查（未触发 get_config）
        assert mgr._config_dict_hash is None or mgr._config_dict_hash != hash_with_override
        # 触发 get_config 重建后 hash 应更新为新值
        mgr.get_config()
        assert mgr._config_dict_hash is not None
        assert mgr._config_dict_hash != hash_with_override
