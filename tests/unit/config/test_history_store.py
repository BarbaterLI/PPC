"""配置版本历史（ConfigHistoryStore）测试。"""

from __future__ import annotations

from src.config.manager_watch import ConfigHistoryStore
from src.config.schemas import PPCConfig


def test_version_numbers_unique_and_monotonic_after_trim():
    """裁剪到 ``max_versions`` 后 ``save_version`` 仍应返回单调递增、不重复的版本号。

    Bug: ``save_version`` 用 ``version = len(self._versions) + 1``。列表被裁剪到
    ``max_versions`` 后 ``len`` 不再等于历史最高版本号，于是版本号开始**重复**：
    max=3 时连存 6 次返回 ``[1,2,3,4,4,4]``，且存储列表出现重复条目——
    ``get_version`` / ``rollback`` 按版本号查找变得歧义。
    修复: 改用单调递增计数器 ``_next_version``。
    """
    store = ConfigHistoryStore(max_versions=3)
    cfg = PPCConfig()

    returned = [store.save_version(cfg) for _ in range(6)]

    assert returned == [1, 2, 3, 4, 5, 6], f"版本号应单调递增不重复: {returned}"

    stored = [v for v, _c, _ts in store._versions]
    assert len(stored) == len(set(stored)), f"存储的版本号重复: {stored}"


def test_rollback_returns_correct_config_after_trim():
    """裁剪后 get_version 仍能按唯一版本号取回正确配置。"""
    store = ConfigHistoryStore(max_versions=5)
    for i in range(8):
        cfg = PPCConfig()
        cfg.tts.concurrency = i + 1
        store.save_version(cfg)

    # 存 8 个（裁到 5）：现存版本应为 4..8，且 concurrency 各不相同
    stored = [v for v, _c, _ts in store._versions]
    assert sorted(stored) == [4, 5, 6, 7, 8], f"现存版本号: {stored}"

    got = store.get_version(7)
    assert got is not None
    assert got.tts.concurrency == 7
