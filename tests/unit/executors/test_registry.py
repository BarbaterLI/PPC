"""Task 23 回归测试: ExecutorRegistry 读方法加锁。

覆盖：
* 读方法 (``get`` / ``get_all`` / ``get_by_tag`` / ``has``) 基本功能正确
* 并发 ``register`` + ``get`` 不崩溃且最终状态一致
"""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from src.executors.base import BaseExecutor
from src.executors.registry import ExecutorRegistry

# ---------------------------------------------------------------------------
# 测试用 BaseExecutor 子类
# ---------------------------------------------------------------------------


class _DummyExecutor(BaseExecutor):
    """最小可注册的 BaseExecutor 子类。"""

    executor_name = "dummy"

    async def initialize(self):
        self._initialized = True

    async def execute(self, input_path: Path, output_path: Path):
        return None


class _AnotherExecutor(BaseExecutor):
    """第二个测试用执行器，用于标签查询。"""

    executor_name = "another"

    async def initialize(self):
        self._initialized = True

    async def execute(self, input_path: Path, output_path: Path):
        return None


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def registry() -> ExecutorRegistry:
    return ExecutorRegistry()


# ---------------------------------------------------------------------------
# 读方法基本功能
# ---------------------------------------------------------------------------


class TestRegistryReadMethods:
    def test_get_returns_none_for_missing(self, registry: ExecutorRegistry) -> None:
        assert registry.get("missing") is None

    def test_get_returns_metadata_after_register(self, registry: ExecutorRegistry) -> None:
        registry.register("dummy", _DummyExecutor, tags=["tts"])
        meta = registry.get("dummy")
        assert meta is not None
        assert meta.name == "dummy"
        assert meta.executor_class is _DummyExecutor
        assert meta.tags == ["tts"]

    def test_get_all_returns_copy(self, registry: ExecutorRegistry) -> None:
        registry.register("dummy", _DummyExecutor)
        all_executors = registry.get_all()
        assert "dummy" in all_executors
        # 返回的是副本，修改不影响内部状态
        all_executors.clear()
        assert registry.get("dummy") is not None

    def test_get_by_tag_filters_correctly(self, registry: ExecutorRegistry) -> None:
        registry.register("dummy", _DummyExecutor, tags=["tts", "audio"])
        registry.register("another", _AnotherExecutor, tags=["audio"])
        tts_only = registry.get_by_tag("tts")
        assert len(tts_only) == 1
        assert tts_only[0].name == "dummy"
        audio_all = registry.get_by_tag("audio")
        assert {m.name for m in audio_all} == {"dummy", "another"}
        assert registry.get_by_tag("nonexistent") == []

    def test_has_returns_true_false_correctly(self, registry: ExecutorRegistry) -> None:
        assert registry.has("dummy") is False
        registry.register("dummy", _DummyExecutor)
        assert registry.has("dummy") is True


# ---------------------------------------------------------------------------
# 并发 register + get 一致性
# ---------------------------------------------------------------------------


class TestRegistryConcurrentReadSafety:
    def test_concurrent_register_and_get_no_crash(self, registry: ExecutorRegistry) -> None:
        """多线程并发 register / get / has / get_all / get_by_tag 不应崩溃，
        且最终状态一致（所有写入都可见）。"""
        names = [f"exec_{i}" for i in range(50)]
        errors: list[Exception] = []
        barrier = threading.Barrier(5)

        def writer():
            try:
                barrier.wait()
                for name in names:
                    # force=True 避免重复注册抛 ValueError（本测试聚焦并发读写安全，
                    # 而非重复注册检测）
                    registry.register(name, _DummyExecutor, tags=["concurrent"], force=True)
            except Exception as e:  # noqa: BLE001
                errors.append(e)

        def reader():
            try:
                barrier.wait()
                for _ in range(2000):
                    # 多种读操作混合，验证不会抛异常
                    registry.get("exec_0")
                    registry.has("exec_25")
                    registry.get_all()
                    registry.get_by_tag("concurrent")
            except Exception as e:  # noqa: BLE001
                errors.append(e)

        threads = [threading.Thread(target=writer) for _ in range(2)] + [
            threading.Thread(target=reader) for _ in range(3)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"并发读写过程中抛出异常: {errors}"

        # 最终一致性：所有注册的名字都应可见
        all_meta = registry.get_all()
        for name in names:
            assert name in all_meta, f"缺失已注册的执行器: {name}"
            assert registry.has(name) is True
            assert registry.get(name) is not None

    def test_concurrent_register_unregister_get_consistent(self, registry: ExecutorRegistry) -> None:
        """并发 register / unregister / get 混合：不应崩溃。

        注意：``has`` 与 ``get`` 是两次独立的锁获取，组合并非原子，
        因此本测试只验证单次读操作返回的是合法快照（不抛异常、类型正确），
        不在两次读之间强加一致性，避免 TOCTOU 误报。
        """
        errors: list[Exception] = []
        stop = threading.Event()
        barrier = threading.Barrier(3)

        def registrar():
            try:
                barrier.wait()
                i = 0
                while not stop.is_set():
                    name = f"cycle_{i % 20}"
                    registry.register(name, _DummyExecutor, force=True)
                    i += 1
            except Exception as e:  # noqa: BLE001
                errors.append(e)

        def unregistrar():
            try:
                barrier.wait()
                i = 0
                while not stop.is_set():
                    name = f"cycle_{i % 20}"
                    registry.unregister(name)
                    i += 1
            except Exception as e:  # noqa: BLE001
                errors.append(e)

        def reader():
            try:
                barrier.wait()
                while not stop.is_set():
                    # 每次单读操作都必须返回合法类型（None 或 ExecutorMetadata），
                    # 不应抛异常或返回半构造对象。
                    name = "cycle_5"
                    get_result = registry.get(name)
                    assert get_result is None or get_result.name == name
                    has_result = registry.has(name)
                    assert isinstance(has_result, bool)
                    # get_all 返回的字典必须可遍历而不抛 RuntimeError（字典在遍历中被修改）
                    snapshot = registry.get_all()
                    for meta in snapshot.values():
                        assert meta.name is not None
            except Exception as e:  # noqa: BLE001
                errors.append(e)

        threads = [
            threading.Thread(target=registrar),
            threading.Thread(target=unregistrar),
            threading.Thread(target=reader),
        ]
        for t in threads:
            t.start()

        # 让线程运行一小段时间
        threading.Event().wait(0.5)
        stop.set()
        for t in threads:
            t.join()

        assert errors == [], f"并发 register/unregister/get 抛出异常: {errors}"
