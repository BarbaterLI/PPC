"""UA health filter / UA 健康过滤器.

Split out of the former monolithic ``edge_tts_client`` module.
"""

from __future__ import annotations

import logging
import time

logger = logging.getLogger(__name__)


class UAHealthFilter:
    """UA 健康过滤器 / User-Agent health filter.

    Tracks UA health status with two distinct failure paths:

    - **403 (explicit UA rejection)**: marked unhealthy immediately —
      Edge TTS explicitly rejected this UA, no point retrying.
    - **NoAudioReceived (transient)**: NOT marked immediately. Instead,
      a per-UA consecutive failure counter is incremented. Only when
      the counter reaches ``no_audio_threshold`` (default 3) is the UA
      marked unhealthy. A successful stream resets the counter.

      This avoids over-aggressive marking because ``NoAudioReceived``
      is a common transient failure under high load — a single
      occurrence that succeeds on retry should not penalize the UA.
      Only repeated failures (until the conversion exhausts retries)
      indicate the UA itself is problematic.

    This filter is shared across all streams from the same
    client instance, so a UA that fails on one
    stream is not retried on concurrent streams.
    """

    def __init__(
        self,
        cooldown_seconds: float = 300.0,
        no_audio_threshold: int = 3,
    ) -> None:
        self._cooldown = cooldown_seconds
        self._no_audio_threshold = no_audio_threshold
        self._unhealthy: dict[str, float] = {}  # ua → time.monotonic() when marked
        # Per-UA consecutive NoAudioReceived count. Reset on success.
        self._no_audio_counts: dict[str, int] = {}

    def is_healthy(self, ua: str) -> bool:
        """Return True if *ua* is healthy (not marked or cooldown expired)."""
        marked_at = self._unhealthy.get(ua)
        if marked_at is None:
            return True
        if time.monotonic() - marked_at > self._cooldown:
            del self._unhealthy[ua]
            return True
        return False

    def mark_unhealthy(self, ua: str) -> None:
        """Mark *ua* as unhealthy for the cooldown period."""
        self._unhealthy[ua] = time.monotonic()
        # Clear the NoAudio counter since the UA is now in cooldown.
        self._no_audio_counts.pop(ua, None)
        logger.warning(
            "UA 已标记为不健康 / UA marked unhealthy (cooldown=%.0fs): %s",
            self._cooldown,
            ua[:60] + "..." if len(ua) > 60 else ua,
        )

    def unmark_unhealthy(self, ua: str) -> None:
        """Revoke an unhealthy mark — used after UA switch verification fails.

        When a UA hits the NoAudioReceived threshold and is marked
        unhealthy, the engine verifies with other UAs. If **all** other
        UAs also fail with NoAudioReceived, the failure is a server-side
        issue — not UA-specific — so the original UA's mark is revoked
        via this method.
        """
        self._unhealthy.pop(ua, None)
        self._no_audio_counts.pop(ua, None)

    def record_no_audio(self, ua: str) -> bool:
        """Record a NoAudioReceived failure for *ua*.

        Increments the per-UA consecutive counter. When the counter
        reaches ``no_audio_threshold``, the UA is marked unhealthy
        (via :meth:`mark_unhealthy`) and the counter is reset.

        Returns ``True`` if the UA was marked unhealthy on this call.
        """
        count = self._no_audio_counts.get(ua, 0) + 1
        if count >= self._no_audio_threshold:
            self.mark_unhealthy(ua)
            return True
        self._no_audio_counts[ua] = count
        logger.debug(
            "UA NoAudioReceived 计数 / count %d/%d: %s",
            count,
            self._no_audio_threshold,
            ua[:60] + "..." if len(ua) > 60 else ua,
        )
        return False

    def record_success(self, ua: str) -> None:
        """Reset the NoAudioReceived counter for *ua* on successful stream.

        A single success means the UA is working — any prior
        NoAudioReceived was transient, not UA-specific.
        """
        self._no_audio_counts.pop(ua, None)

    def filter_healthy(self, uas: list[str]) -> list[str]:
        """Return only healthy UAs from *uas*."""
        return [ua for ua in uas if self.is_healthy(ua)]

    @staticmethod
    def is_ua_failure(exc: BaseException) -> bool:
        """Return True if *exc* is a 403 (explicit UA rejection).

        403 errors warrant **immediate** marking — Edge TTS explicitly
        rejected the request based on the UA.
        """
        # aiohttp.ClientResponseError has a ``status`` attribute
        status = getattr(exc, "status", None)
        if status == 403:
            return True
        # Fallback: check message for 403
        if "403" in str(exc):
            return True
        return False

    @staticmethod
    def is_no_audio_failure(exc: BaseException) -> bool:
        """Return True if *exc* is a ``NoAudioReceived`` error.

        NoAudioReceived is a transient failure that is **not** marked
        immediately — instead, :meth:`record_no_audio` increments a
        per-UA counter and only marks unhealthy after
        ``no_audio_threshold`` consecutive failures.
        """
        return type(exc).__name__ == "NoAudioReceived"
