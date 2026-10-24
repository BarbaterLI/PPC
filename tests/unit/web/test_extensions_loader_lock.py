"""V10：``get_loader(ensure_loaded=True)`` 首次并发请求只加载一次。

审计待验证项 V10：``_loader_loaded`` 是裸 bool 的 check-then-act，Web 以
``threaded=True`` 运行时，多个首次请求会同时判定「未加载」并重复执行
``load_all_extensions``（虽然幂等，但会产生重复目录扫描与竞态窗口）。

修复后用 ``RLock`` 做 double-checked locking，本测试用线程 barrier 把 N 个
线程卡在同一起跑线，断言实际加载只发生一次。
"""

from __future__ import annotations

import threading

import pytest

from src.web.api import extensions as extensions_api


class _CountingLoader:
    """记录 ``load_all_extensions`` 调用次数的假 loader。"""

    def __init__(self) -> None:
        self.calls = 0
        self._lock = threading.Lock()

    async def load_all_extensions(self, force: bool = False):
        with self._lock:
            self.calls += 1
        # 拉长加载窗口，放大旧实现的竞态（无锁时必然重复加载）
        import time

        time.sleep(0.05)
        return {}

    def get_loaded_extensions(self):
        return {}


@pytest.fixture(autouse=True)
def _reset_loader_state(monkeypatch: pytest.MonkeyPatch):
    """每个测试重置模块级单例，并让加载走假 loader / 启用扩展。"""
    loader = _CountingLoader()
    monkeypatch.setattr(extensions_api, "_loader", None, raising=False)
    monkeypatch.setattr(extensions_api, "_loader_loaded", False, raising=False)

    import src.extensions.loader as loader_mod

    monkeypatch.setattr(loader_mod, "build_loader_from_config", lambda: loader)
    monkeypatch.setattr(loader_mod, "extensions_enabled_by_config", lambda: True)
    yield loader


def test_concurrent_first_requests_load_once(_reset_loader_state) -> None:
    loader = _reset_loader_state
    n = 8
    barrier = threading.Barrier(n)
    results: list[object] = []
    errors: list[BaseException] = []

    def worker() -> None:
        try:
            barrier.wait(timeout=5)  # 同时冲过「未加载」判断
            results.append(extensions_api.get_loader(ensure_loaded=True))
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert not errors, errors
    assert len(results) == n
    # 全部拿到同一个 loader 实例
    assert all(r is loader for r in results)
    # 只真正加载一次（旧实现会 8 次）
    assert loader.calls == 1


def test_second_call_does_not_reload(_reset_loader_state) -> None:
    loader = _reset_loader_state

    extensions_api.get_loader(ensure_loaded=True)
    extensions_api.get_loader(ensure_loaded=True)

    assert loader.calls == 1


def test_ensure_loaded_false_never_loads(_reset_loader_state) -> None:
    loader = _reset_loader_state

    extensions_api.get_loader()
    extensions_api.get_loader()

    assert loader.calls == 0
