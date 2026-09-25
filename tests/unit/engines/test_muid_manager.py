"""Unit tests for :class:`src.engines.muid_manager.MuidManager` persistence.

Covers the single-writer invariant: concurrent ``get_muid()`` on a fresh
manager must not corrupt or thrash the persisted MUID file (mutex + atomic
replace).
"""

from __future__ import annotations

import threading

from src.engines.muid_manager import MuidManager


def test_concurrent_get_muid_persists_valid_file(tmp_path) -> None:
    target = tmp_path / "edge_muid"
    manager = MuidManager(file_path=str(target), rotate_hours=24.0, persist=True)

    results: list[str] = []
    errors: list[BaseException] = []
    barrier = threading.Barrier(8)

    def _get() -> None:
        try:
            barrier.wait(timeout=5)
            results.append(manager.get_muid())
        except BaseException as e:  # noqa: BLE001 - let the test surface first failure
            errors.append(e)

    threads = [threading.Thread(target=_get) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert errors == []
    assert len(results) == 8
    # Non-persist falls back to random, but persist path must return one value.
    assert len(set(results)) == 1

    persisted = target.read_text(encoding="utf-8").strip()
    assert persisted == results[0]
    # No leftover temp file after atomic replace.
    assert not (target.with_suffix(".tmp")).exists() or not (tmp_path / "edge_muid.tmp").exists()


def test_invalidate_then_regenerate(tmp_path) -> None:
    target = tmp_path / "edge_muid"
    manager = MuidManager(file_path=str(target), rotate_hours=24.0, persist=True)
    first = manager.get_muid()
    assert target.exists()

    manager.invalidate()
    assert not target.exists()

    second = manager.get_muid()
    assert second != first
    assert target.read_text(encoding="utf-8").strip() == second

