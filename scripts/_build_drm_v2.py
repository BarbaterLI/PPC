"""构建 DRM v2 移植后的 drm.py：算法区从 7.3.1 逐字节搬运，接入实例化 ClockSkew。"""

from pathlib import Path

SRC = Path(r"D:/PPCs/12.0.0/beta/.dev/edge-tts-src/edge_tts-7.3.1/src/edge_tts/drm.py")
DST = Path(r"D:/PPCs/12.0.0/beta/PPC/src/engines/edge_tts/drm.py")

src_lines = SRC.read_text(encoding="utf-8").splitlines(keepends=True)

# 算法区：# FIPS 180-4: (第 21 行，1-based) 到 _token_hash 结束（第 459 行）
start = next(i for i, l in enumerate(src_lines) if l.startswith("# FIPS 180-4:"))
end = next(i for i, l in enumerate(src_lines) if l.startswith("def generate_sec_ms_gec_v2"))
algo = "".join(src_lines[start:end]).rstrip("\n")

header = '''"""DRM module is used to handle DRM operations with clock skew correction.
Currently the only DRM operation is generating the Sec-MS-GEC token value
used in all API requests to Microsoft Edge's online text-to-speech service.

**Sec-MS-GEC v2**（2026-10，随上游 edge-tts 7.3.1 启用）：服务端已拒绝旧的
v1 token（纯 SHA256 hex），握手一律返回 403。v2 token 形如
``<16 hex CS64>-<64 hex SHA256>``（共 81 字符）。

Clock skew state lives in an injectable :class:`ClockSkew` instance instead
of a class-level mutable attribute. 注意：上游 7.3.1 的
``generate_sec_ms_gec_v2`` 走 ``DRM.get_unix_timestamp()`` 类方法；本副本已把
时钟偏移改为实例持有，故 token 生成必须取实例时间戳，否则 403 后的自动校时
（``handle_client_response_error``）会静默失效。
"""

import secrets
import struct
from datetime import datetime as dt
from datetime import timezone as tz
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import aiohttp

from .constants import TRUSTED_CLIENT_TOKEN
from .exceptions import SkewAdjustmentError

WIN_EPOCH = 11644473600
S_TO_NS = 1e9

# ---------------------------------------------------------------------------
# 以下 SHA-256 / CS64 实现逐字节取自上游 edge-tts 7.3.1（drm.py），
# 保留原样以便与上游 diff；请勿改写，回归由 tests/unit/engines/test_drm_v2.py
# 的已知答案用例锁住。
# ---------------------------------------------------------------------------
'''

tail = '''

# ---------------------------------------------------------------------------
# Sec-MS-GEC v2 入口
# ---------------------------------------------------------------------------


def _sec_ms_gec_from_timestamp(ticks_unix: float) -> str:
    """由（已含时钟偏移校正的）Unix 时间戳生成 v2 Sec-MS-GEC token。

    ticks 计算沿用 v1：Windows file time、5 分钟向下取整；差别只在最后一步
    由「单段 SHA256」换成「CS64 + SHA256」的复合摘要。
    """
    ticks = ticks_unix + WIN_EPOCH
    ticks -= ticks % 300
    ticks *= S_TO_NS / 100
    str_to_hash = f"{ticks:.0f}{TRUSTED_CLIENT_TOKEN}".encode("ascii")
    cs64, sha256 = _token_hash(str_to_hash)
    return f"{cs64}-{sha256}"


def generate_sec_ms_gec_v2(unix_timestamp: Optional[float] = None) -> str:
    """Generate a v2 Sec-MS-GEC token value.

    Args:
        unix_timestamp: Value to use instead of the current time (for testing).

    Returns:
        str: The generated v2 token value, ``"<16 hex>-<64 hex>"``.
    """
    if unix_timestamp is None:
        # 共享实例：403 驱动的时钟偏移校正在进程内持续生效。
        unix_timestamp = DEFAULT_DRM.get_unix_timestamp()
    return _sec_ms_gec_from_timestamp(unix_timestamp)


class ClockSkew:
    """Clock skew state, injectable into :class:`DRM`.

    Holds the accumulated skew (in seconds) between the local system
    clock and the server clock, as learned from 403 responses.
    """

    def __init__(self) -> None:
        self._seconds: float = 0.0

    @property
    def seconds(self) -> float:
        return self._seconds

    def adjust(self, skew_seconds: float) -> None:
        """Adjust the clock skew in seconds in case the system clock is off."""
        self._seconds += skew_seconds

    def get_unix_timestamp(self) -> float:
        """Gets the current timestamp in Unix format with clock skew correction."""
        return dt.now(tz.utc).timestamp() + self._seconds


# Shared default skew state: 403-driven skew corrections learned by one
# request keep applying to all subsequent requests in this process
# (same behavior as the former class-level attribute).
DEFAULT_CLOCK_SKEW = ClockSkew()


class DRM:
    """
    Class to handle DRM operations with clock skew correction.
    """

    def __init__(self, clock_skew: Optional[ClockSkew] = None) -> None:
        self._clock_skew = clock_skew if clock_skew is not None else DEFAULT_CLOCK_SKEW

    def adj_clock_skew_seconds(self, skew_seconds: float) -> None:
        """Adjust the clock skew in seconds in case the system clock is off."""
        self._clock_skew.adjust(skew_seconds)

    def get_unix_timestamp(self) -> float:
        """Current timestamp in Unix format with clock skew correction."""
        return self._clock_skew.get_unix_timestamp()

    @staticmethod
    def parse_rfc2616_date(date: str) -> Optional[float]:
        """Parse an RFC 2616 date string into a Unix timestamp."""
        try:
            return (
                dt.strptime(date, "%a, %d %b %Y %H:%M:%S %Z")
                .replace(tzinfo=tz.utc)
                .timestamp()
            )
        except ValueError:
            return None

    def handle_client_response_error(self, e: aiohttp.ClientResponseError) -> None:
        """Adjust clock skew from the server date on a failed response.

        Raises :class:`SkewAdjustmentError` if the server date is missing
        or cannot be parsed.
        """
        if e.headers is None:
            raise SkewAdjustmentError("No server date in headers.") from e
        server_date: Optional[str] = e.headers.get("Date", None)
        if server_date is None or not isinstance(server_date, str):
            raise SkewAdjustmentError("No server date in headers.") from e
        server_date_parsed: Optional[float] = DRM.parse_rfc2616_date(server_date)
        if server_date_parsed is None or not isinstance(server_date_parsed, float):
            raise SkewAdjustmentError(
                f"Failed to parse server date: {server_date}"
            ) from e
        client_date = self.get_unix_timestamp()
        self.adj_clock_skew_seconds(server_date_parsed - client_date)

    def generate_sec_ms_gec(self) -> str:
        """Generate the Sec-MS-GEC token value (**v2** format).

        v2 = ``<16 hex CS64>-<64 hex SHA256>`` (81 chars). 服务端自 2026-10 起
        拒绝 v1（纯 SHA256 hex）token，握手返回 403，故此处必须走 v2。

        时间戳取自 **实例** 的 :meth:`get_unix_timestamp`（含注入的时钟偏移），
        而非上游的类方法 —— 否则 403 自动校时会失效。

        See Also:
            https://github.com/rany2/edge-tts/issues/290#issuecomment-2464956570
        """
        return _sec_ms_gec_from_timestamp(self.get_unix_timestamp())

    @staticmethod
    def generate_muid() -> str:
        """Generate a random MUID."""
        return secrets.token_hex(16).upper()

    @staticmethod
    def headers_with_muid(headers: Dict[str, str], muid: Optional[str] = None) -> Dict[str, str]:
        """Return a copy of *headers* with the MUID cookie added.

        Passing ``muid`` avoids the per-call ``DRM.generate_muid`` mutation that
        causes race conditions under concurrent access.
        """
        combined_headers = headers.copy()
        assert "Cookie" not in combined_headers
        muid_value = muid if muid is not None else DRM.generate_muid()
        combined_headers["Cookie"] = f"muid={muid_value};"
        return combined_headers


# Shared default DRM instance (uses the shared DEFAULT_CLOCK_SKEW).
DEFAULT_DRM = DRM()
'''

DST.write_text(header + algo + tail, encoding="utf-8")
print(f"写入 {DST}，共 {len(DST.read_text(encoding='utf-8').splitlines())} 行")
print(f"算法区搬运 {len(algo.splitlines())} 行（来源 {SRC}）")
