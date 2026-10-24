"""P2-9: ``ConfigNotifier.notify`` 迭代期间可被监听器修改。

触发条件：某监听器在 ``on_config_change`` 内部调用 ``add_listener`` /
``remove_listener`` 修改监听器表。
后果（修复前）：``RuntimeError: dictionary changed size during iteration``，
异常未被捕获并冒泡到 ``ConfigManager.set()``，导致本次配置变更整体失败。
"""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from pathlib import Path

import yaml

from src.config.manager_core import ConfigManager
from src.config.manager_notify import ConfigNotifier
from src.config.manager_state import ConfigChangeEvent


class _LockOnlyStore:
    """仅提供 ``_lock`` 的最小 ConfigStore 替身（ConfigNotifier 只依赖该属性）。"""

    def __init__(self) -> None:
        self._lock = threading.RLock()


class _Recorder:
    def __init__(self, name: str, calls: list[str]) -> None:
        self.name = name
        self.calls = calls

    def on_config_change(self, event: ConfigChangeEvent) -> None:
        self.calls.append(self.name)


class _Exploding:
    def __init__(self, exc: Exception) -> None:
        self.exc = exc

    def on_config_change(self, event: ConfigChangeEvent) -> None:
        raise self.exc


class _SelfRemovingAndAdding:
    """回调中移除自己并新增另一个监听器。"""

    def __init__(self, notifier: ConfigNotifier, calls: list[str], new_listener: _Recorder) -> None:
        self._notifier = notifier
        self._calls = calls
        self._new_listener = new_listener

    def on_config_change(self, event: ConfigChangeEvent) -> None:
        self._calls.append("self_removing")
        self._notifier.remove_listener(self)
        self._notifier.add_listener("*", self._new_listener)


def _event(key: str = "tts.voice") -> ConfigChangeEvent:
    return ConfigChangeEvent(
        key=key,
        old_value="a",
        new_value="b",
        timestamp=datetime.now(timezone.utc),
    )


def test_listener_mutating_registry_during_notify_does_not_raise() -> None:
    notifier = ConfigNotifier(_LockOnlyStore())
    calls: list[str] = []
    added = _Recorder("added", calls)
    mutator = _SelfRemovingAndAdding(notifier, calls, added)
    bystander = _Recorder("bystander", calls)

    notifier.add_listener("tts.*", mutator)
    notifier.add_listener("*", bystander)

    notifier.notify(_event())  # 修复前抛 RuntimeError

    # 快照语义：本轮其余监听器仍被调用，新增的 B 只在后续事件生效
    assert calls == ["self_removing", "bystander"]
    # "*" 下原有 bystander，回调中又追加了 added；快照不影响登记结果
    assert notifier._listeners.get("*") == [bystander, added]
    # 移除自身只影响其注册的 "tts.*"，该模式已空并被删除
    assert "tts.*" not in notifier._listeners


def test_listener_adding_new_pattern_during_notify() -> None:
    notifier = ConfigNotifier(_LockOnlyStore())
    calls: list[str] = []
    late = _Recorder("late", calls)

    class _Adder:
        def on_config_change(self, event: ConfigChangeEvent) -> None:
            calls.append("adder")
            notifier.add_listener("tts.voice", late)

    notifier.add_listener("*", _Adder())
    notifier.notify(_event())

    assert calls == ["adder"]
    assert "tts.voice" in notifier._listeners


def test_failing_listener_is_isolated(caplog) -> None:
    notifier = ConfigNotifier(_LockOnlyStore())
    calls: list[str] = []
    notifier.add_listener("*", _Recorder("ok", calls))
    notifier.add_listener("*", _Exploding(RuntimeError("boom")))
    notifier.add_listener("*", _Recorder("after", calls))

    with caplog.at_level("WARNING", logger="src.config.manager_notify"):
        notifier.notify(_event())  # 不应向外抛

    assert calls == ["ok", "after"]
    assert any("监听器处理失败" in record.message for record in caplog.records)


def test_config_manager_set_succeeds_when_listener_raises(tmp_path: Path) -> None:
    """监听器异常不得中断 ``ConfigManager.set()``。"""
    (tmp_path / "config.yml").write_text(
        yaml.dump(
            {"version": ConfigManager.CONFIG_VERSION, "tts": {"voice": "zh-CN-XiaoxiaoNeural", "preset": "balanced"}},
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    manager = ConfigManager(config_dir=tmp_path)
    manager.get_config()

    calls: list[str] = []
    manager.add_listener("*", _Exploding(ValueError("listener boom")))
    manager.add_listener("*", _Recorder("survivor", calls))

    assert manager.set("tts.voice", "zh-CN-YunxiNeural", persist=False) is True
    assert calls == ["survivor"]


def test_notify_snapshot_allows_concurrent_add_remove() -> None:
    """多线程并发增删监听器时 notify 不应因字典变动而失败。"""
    notifier = ConfigNotifier(_LockOnlyStore())
    errors: list[BaseException] = []
    calls: list[str] = []

    def _notify_loop() -> None:
        try:
            for _ in range(200):
                notifier.notify(_event())
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    def _mutate_loop(offset: int) -> None:
        try:
            for i in range(200):
                listener = _Recorder(f"l{offset}-{i}", calls)
                notifier.add_listener("*", listener)
                notifier.remove_listener(listener)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=_notify_loop), threading.Thread(target=_mutate_loop, args=(0,))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert errors == []


def test_notify_matches_event_key() -> None:
    notifier = ConfigNotifier(_LockOnlyStore())
    calls: list[str] = []
    notifier.add_listener("tts.voice", _Recorder("voice", calls))
    notifier.add_listener("audio", _Recorder("audio", calls))

    notifier.notify(_event("tts.voice"))
    assert calls == ["voice"]
