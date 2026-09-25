"""ConfigManager 配置缓存优化测试。

对应 spec ``optimize-config-and-edge-selection`` Task 6：在内容未变化时跳过
``PPCConfig`` 重建 / Pydantic 校验，mtime 与 temp_overrides 仍能强制重建。
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import yaml

from src.config.manager_core import ConfigManager


def _write_config(path: Path, voice: str = "zh-CN-XiaoxiaoNeural") -> None:
    """向 ``path`` 写入一份最小的可用 ``config.yml``。"""
    data = {
        "version": ConfigManager.CONFIG_VERSION,
        "tts": {"voice": voice, "preset": "balanced"},
    }
    with path.open("w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True, default_flow_style=False)


def test_cache_hit_returns_same_instance(tmp_path: Path) -> None:
    """相同配置多次加载应返回同一个 PPCConfig 实例（哈希缓存命中）。"""
    _write_config(tmp_path / "config.yml")

    mgr = ConfigManager(config_dir=tmp_path)
    first = mgr.get_config()
    cached_hash = mgr._config_dict_hash
    assert cached_hash is not None, "首次加载应建立哈希缓存"

    # 让 _check_cache 判定 TTL 过期，从而再次触发 _load()
    mgr._cache_time = 0
    second = mgr.get_config()

    assert second is first, "配置未变化时应返回同一实例（缓存命中）"
    assert mgr._config_dict_hash == cached_hash, "缓存命中不应改变哈希"


def test_mtime_change_rebuilds(tmp_path: Path) -> None:
    """文件 mtime 变化（内容不变）时，_load 应重建 PPCConfig（新实例）。"""
    config_path = tmp_path / "config.yml"
    _write_config(config_path, voice="zh-CN-XiaoxiaoNeural")

    mgr = ConfigManager(config_dir=tmp_path)
    first = mgr.get_config()
    first_voice = first.tts.voice

    # 内容保持不变，仅刷新 mtime
    time.sleep(0.1)
    os.utime(config_path, None)

    mgr._load()
    second = mgr.get_config()

    assert second is not first, "mtime 变化后应重建为新实例"
    assert second.tts.voice == first_voice, "内容未变化，值应保持一致"


def test_temp_override_change_rebuilds(tmp_path: Path) -> None:
    """temp_overrides 变化后，下次 _load 应重建 PPCConfig（新实例）。"""
    _write_config(tmp_path / "config.yml")

    mgr = ConfigManager(config_dir=tmp_path)
    first = mgr.get_config()
    assert mgr._config_dict_hash is not None, "首次加载应建立哈希缓存"

    # set_temp 会失效哈希缓存；再显式 _load 触发基于新哈希的重建
    mgr.set_temp("tts.voice", "zh-CN-YunxiNeural")
    assert mgr._config_dict_hash is None, "set_temp 应失效哈希缓存"
    mgr._load()

    second = mgr.get_config()
    assert second is not first, "temp_overrides 变化后应重建为新实例"
    assert mgr._config_dict_hash is not None, "重建后应更新哈希缓存"
