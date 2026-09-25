"""Unit tests for ``src.events.event_bus``.

Covers:
- ``fire_and_forget`` publish mode does not block on slow async handlers
- constructing ``EventBus`` / ``get_event_bus()`` has no side effects
  (no background thread, no ``atexit`` registration)
- ``atexit`` hook is registered by the explicit ``start()`` lifecycle method
- the background async loop starts lazily on first ``publish_async``
- Backward-compatible blocking behaviour is preserved by default
"""

from __future__ import annotations

import asyncio
import threading
import time
from unittest import mock

from src.events import event_bus as event_bus_module
from src.events.event_bus import EventBus, TaskStartedEvent

# -----------------------------------------------------------------------------
# fire_and_forget
# -----------------------------------------------------------------------------


class TestFireAndForget:
    def test_fire_and_forget_does_not_block_on_slow_async_handler(self):
        """publish(fire_and_forget=True) must return well before the handler finishes."""
        bus = EventBus()
        try:
            completed = threading.Event()

            async def slow_handler(_event: TaskStartedEvent) -> None:
                await asyncio.sleep(2)
                completed.set()

            bus.subscribe(TaskStartedEvent, slow_handler)

            start = time.monotonic()
            bus.publish(TaskStartedEvent(task_id="t1"), fire_and_forget=True)
            elapsed = time.monotonic() - start

            # Without fire_and_forget this would take ~2s; with it, well under 1s.
            assert elapsed < 1.0, f"publish took {elapsed:.3f}s, expected < 1.0s"
            # Handler still runs on the background loop.
            assert completed.wait(timeout=5), "slow handler did not complete"
        finally:
            bus.shutdown()

    def test_default_publish_blocks_on_async_handler(self):
        """Default publish() must wait for async handlers (backward compat)."""
        bus = EventBus()
        try:
            completed = threading.Event()

            async def handler(_event: TaskStartedEvent) -> None:
                await asyncio.sleep(0.2)
                completed.set()

            bus.subscribe(TaskStartedEvent, handler)

            start = time.monotonic()
            bus.publish(TaskStartedEvent(task_id="t2"))
            elapsed = time.monotonic() - start

            # Should have blocked ~0.2s for the handler.
            assert elapsed >= 0.15, f"publish returned in {elapsed:.3f}s, expected blocking"
            assert completed.is_set(), "handler should have completed before publish returned"
        finally:
            bus.shutdown()

    def test_fire_and_forget_with_sync_handler_unaffected(self):
        """fire_and_forget has no effect on synchronous handlers (still invoked inline)."""
        bus = EventBus()
        try:
            called = threading.Event()

            def sync_handler(_event: TaskStartedEvent) -> None:
                called.set()

            bus.subscribe(TaskStartedEvent, sync_handler)
            bus.publish(TaskStartedEvent(task_id="t3"), fire_and_forget=True)
            assert called.is_set()
        finally:
            bus.shutdown()


# -----------------------------------------------------------------------------
# lazy lifecycle (background thread + atexit)
# -----------------------------------------------------------------------------


class TestLazyLifecycle:
    def test_construction_starts_no_background_thread(self):
        """EventBus() must not create the async loop thread eagerly."""
        bus = EventBus()
        try:
            assert bus._async_loop_thread is None
            assert bus._async_loop is None
        finally:
            bus.shutdown()

    def test_get_event_bus_singleton_without_atexit(self):
        """get_event_bus() lazily creates the singleton without atexit registration."""
        event_bus_module.reset_event_bus()
        try:
            with mock.patch.object(event_bus_module, "atexit") as mock_atexit:
                bus = event_bus_module.get_event_bus()
                mock_atexit.register.assert_not_called()
                # Singleton identity is stable.
                assert event_bus_module.get_event_bus() is bus
                # No background thread either.
                assert bus._async_loop_thread is None
        finally:
            event_bus_module.reset_event_bus()

    def test_start_registers_atexit_exactly_once(self):
        """start() must register shutdown via atexit exactly once (idempotent)."""
        event_bus_module.reset_event_bus()
        try:
            with mock.patch.object(event_bus_module, "atexit") as mock_atexit:
                bus = event_bus_module.get_event_bus()
                bus.start()
                bus.start()  # idempotent: no second registration
                mock_atexit.register.assert_called_once_with(bus.shutdown)
                # start() also (re)uses the background loop.
                assert bus._async_loop_thread is not None
                assert bus._async_loop_thread.is_alive()
        finally:
            event_bus_module.reset_event_bus()

    def test_atexit_unregistered_on_reset(self):
        """reset_event_bus() must unregister the shutdown hook to avoid leaks in tests."""
        event_bus_module.reset_event_bus()

        with mock.patch.object(event_bus_module, "atexit") as mock_atexit:
            bus = event_bus_module.get_event_bus()
            bus.start()
            assert mock_atexit.register.call_count == 1
            event_bus_module.reset_event_bus()
            mock_atexit.unregister.assert_called_once_with(bus.shutdown)

    def test_publish_async_lazily_starts_loop(self):
        """First publish_async must lazily start the background loop."""
        bus = EventBus()
        try:
            assert bus._async_loop_thread is None

            asyncio.run(bus.publish_async(TaskStartedEvent(task_id="lazy")))

            assert bus._async_loop_thread is not None
            assert bus._async_loop_thread.is_alive()
        finally:
            bus.shutdown()

    def test_shutdown_is_idempotent_for_atexit_safety(self):
        """shutdown() must be safe to call multiple times (atexit + explicit)."""
        bus = EventBus()
        bus.start()
        bus.shutdown()
        # Second call should not raise.
        bus.shutdown()
