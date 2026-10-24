"""CommunicateAdapter: wraps backend Communicate with anti-risk-control params.

包装后端 Communicate，注入风控规避参数。Split out of the former
monolithic ``edge_tts_client`` module.
"""

from __future__ import annotations

import asyncio
import logging
import random
from typing import Any

from src.config.schema import AntiRiskControlConfig

from .muid_manager import MuidManager
from .ua_health_filter import UAHealthFilter

logger = logging.getLogger(__name__)


class CommunicateAdapter:
    """包装后端 Communicate，注入风控规避参数 / Wraps backend Communicate with anti-risk-control params.

    Per-request isolation: MUID and User-Agent are computed once per
    ``stream()`` call and injected into the wrapped ``Communicate``
    instance via ``_muid`` / ``_extra_headers`` attributes. This avoids
    the class-level mutation of ``DRM.generate_muid`` and
    ``constants.BASE_HEADERS`` that caused race conditions under
    concurrent streams.
    """

    def __init__(
        self,
        communicate: Any,
        *,
        anti_risk_config: AntiRiskControlConfig | None = None,
        muid_manager: MuidManager | None = None,
        ua_health_filter: UAHealthFilter | None = None,
        exclude_uas: set[str] | None = None,
    ) -> None:
        self._communicate = communicate
        self._config = anti_risk_config
        self._muid_manager = muid_manager
        # UA 切换验证已尝试过的 UA：从候选池中排除，确保每次重试落到
        # 一个真正的新 UA 上（见 ``_retry_on_no_audio``）。
        self._exclude_uas: frozenset[str] = frozenset(exclude_uas or ())
        self._ua_health_filter = ua_health_filter
        # Tracks the UA selected in the most recent ``stream()`` call,
        # so callers (e.g. ``_synthesize_single_voice``) can inspect it
        # for UA-switch verification on NoAudioReceived.
        self._last_selected_ua: str | None = None

    @property
    def last_selected_ua(self) -> str | None:
        """Return the UA used in the most recent ``stream()`` call."""
        return self._last_selected_ua

    async def stream(self):
        """Proxy stream() with jitter injection and per-request MUID/UA injection.

        The inner ``Communicate.stream()`` generator is explicitly
        ``aclose()``d in a ``finally`` block so that ``CancelledError``
        (from ``asyncio.wait_for`` timeout) or ``GeneratorExit`` (from
        explicit ``aclose()`` on retry) properly cleans up the inner
        ``ClientSession`` / WebSocket — the root cause of
        "Session is closed" errors on subsequent streams.
        """
        # Inject jitter before connection
        if self._config and self._config.enabled and self._config.jitter_enabled:
            delay = random.uniform(0, self._config.jitter_max_seconds)
            if delay > 0:
                await asyncio.sleep(delay)

        # Per-request isolation: compute MUID and UA once, inject via
        # instance attributes on the wrapped Communicate. The backend
        # reads ``self._muid`` and ``self._extra_headers`` in
        # ``_stream()`` — no class-level mutation required.
        selected_ua: str | None = None
        if self._config and self._config.enabled:
            if self._muid_manager is not None and self._config.muid_persist:
                try:
                    muid = self._muid_manager.get_muid()
                    setattr(self._communicate, "_muid", muid)
                except Exception as e:  # noqa: BLE001
                    logger.debug("MUID 注入失败 / MUID injection failed: %s", e)
            if self._config.user_agents:
                try:
                    # Filter out unhealthy UAs; fall back to full pool
                    # if all are unhealthy (better to try than to skip).
                    # 先排除本次合成中已验证失败过的 UA（UA 切换验证），
                    # 再按健康状况过滤；若排除后为空（池中 UA 都试过了），
                    # 保留原池以便仍有重试机会。
                    pool = self._config.user_agents
                    if self._exclude_uas:
                        remaining = [
                            ua for ua in pool if ua not in self._exclude_uas
                        ]
                        if remaining:
                            pool = remaining
                    if self._ua_health_filter is not None:
                        healthy = self._ua_health_filter.filter_healthy(pool)
                        if healthy:
                            pool = healthy
                    selected_ua = random.choice(pool)
                    # Merge into any existing extra_headers so we don't
                    # clobber other per-stream overrides.
                    existing = getattr(self._communicate, "_extra_headers", None) or {}
                    existing = dict(existing)
                    existing["User-Agent"] = selected_ua
                    setattr(self._communicate, "_extra_headers", existing)
                except Exception as e:  # noqa: BLE001
                    logger.debug("UA 注入失败 / UA injection failed: %s", e)
        # Record for external inspection (UA switch verification).
        self._last_selected_ua = selected_ua

        inner_gen = self._communicate.stream()
        stream_ok = False
        try:
            async for chunk in inner_gen:
                yield chunk
            stream_ok = True
        except Exception as exc:
            # 403 (explicit UA rejection): mark unhealthy immediately.
            # NoAudioReceived is NOT handled here — it is counted in
            # ``_synthesize_single_voice`` which has the UA-switch
            # verification logic (retry with a different UA, revoke
            # mark if all UAs fail). Handling it here would double-count.
            if selected_ua is not None and self._ua_health_filter is not None:
                if UAHealthFilter.is_ua_failure(exc):
                    self._ua_health_filter.mark_unhealthy(selected_ua)
            raise
        finally:
            # On success, reset the NoAudioReceived counter for this UA
            # — the UA is working, any prior transient failures are
            # cleared so the threshold doesn't accumulate across
            # independent successful conversions.
            if stream_ok and selected_ua is not None and self._ua_health_filter is not None:
                self._ua_health_filter.record_success(selected_ua)
            # Ensure the inner Communicate.stream() generator is always
            # aclosed — whether on normal completion, CancelledError
            # (wait_for timeout), or GeneratorExit (explicit aclose on
            # retry). Without this, the inner ClientSession leaks,
            # contaminating the shared connector pool and causing
            # "Session is closed" errors on subsequent streams.
            aclose = getattr(inner_gen, "aclose", None)
            if aclose is not None:
                try:
                    await aclose()
                except Exception:  # noqa: BLE001
                    pass
