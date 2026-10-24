"""DRM module is used to handle DRM operations with clock skew correction.
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
# FIPS 180-4:
# https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.180-4.pdf
_SHA256_INITIAL = (
    0x6A09E667,
    0xBB67AE85,
    0x3C6EF372,
    0xA54FF53A,
    0x510E527F,
    0x9B05688C,
    0x1F83D9AB,
    0x5BE0CD19,
)
_SHA256_STATE = (
    0xFD9FBEA2,
    0x91243CE7,
    0xBE499094,
    0xCD2E1E0A,
    0x6C006F49,
    0x04BD35AB,
    0xFEB0A7E4,
    0x03EFFBD2,
)
_SHA256_SCHEDULE = (
    0xEB24BF9E,
    0x1558691F,
    0xA0C09EE2,
    0x0CFA32BA,
    0xA622EE8C,
    0xD1DD9491,
    0xC0DD25A0,
    0xF9E70AB6,
    0xF0C31573,
    0x7058CE4E,
    0x7188E606,
    0x00E5AB33,
    0xD258B060,
    0x72766564,
)
_SHA256_ROUND_CONSTANTS = (
    0x428A2F98,
    0x71374491,
    0xB5C0FBCF,
    0xE9B5DBA5,
    0x3956C25B,
    0x59F111F1,
    0x923F82A4,
    0xAB1C5ED5,
    0xD807AA98,
    0x12835B01,
    0x243185BE,
    0x550C7DC3,
    0x72BE5D74,
    0x80DEB1FE,
    0x9BDC06A7,
    0xC19BF174,
    0xE49B69C1,
    0xEFBE4786,
    0x0FC19DC6,
    0x240CA1CC,
    0x2DE92C6F,
    0x4A7484AA,
    0x5CB0A9DC,
    0x76F988DA,
    0x983E5152,
    0xA831C66D,
    0xB00327C8,
    0xBF597FC7,
    0xC6E00BF3,
    0xD5A79147,
    0x06CA6351,
    0x14292967,
    0x27B70A85,
    0x2E1B2138,
    0x4D2C6DFC,
    0x53380D13,
    0x650A7354,
    0x766A0ABB,
    0x81C2C92E,
    0x92722C85,
    0xA2BFE8A1,
    0xA81A664B,
    0xC24B8B70,
    0xC76C51A3,
    0xD192E819,
    0xD6990624,
    0xF40E3585,
    0x106AA070,
    0x19A4C116,
    0x1E376C08,
    0x2748774C,
    0x34B0BCB5,
    0x391C0CB3,
    0x4ED8AA4A,
    0x5B9CCA4F,
    0x682E6FF3,
    0x748F82EE,
    0x78A5636F,
    0x84C87814,
    0x8CC70208,
    0x90BEFFFA,
    0xA4506CEB,
    0xBEF9A3F7,
    0xC67178F2,
)
_SHA256_ADJUSTMENT = 0xC9DC06A7
_SHA256_BLOCK_CONSTANTS = (
    _SHA256_ROUND_CONSTANTS[:14] + (_SHA256_ADJUSTMENT,) + _SHA256_ROUND_CONSTANTS[15:]
)
_SHA256_SIGMA0_ADJUSTMENT = 0x059C0B80
_SHA256_SIGMA1_ADJUSTMENT = 0x000B92C0

_CS64_COEFFICIENTS = (
    (0xE82E91B1, 0xB32700D0),
    (0x44DE24C6, 0x88DFF51D),
    (0x3BEB65BC, 0x29514038),
    (0xC89C5933, 0xB328CBDF),
    (0x8D1EC211, 0xD405B4D5),
    (0x314EF5D5, 0x26E55F3C),
    (0xE92491B1, 0x607801C6),
    (0xF0A5B9D5, 0x82B2C1FB),
    (0x06064CF3, 0xFA51A3DA),
    (0xDA79FA60, 0x207CC464),
    (0xBED20560, 0x3C3C74F3),
    (0x89F9C9F8, 0xD2D14B61),
    (0x35FA38C6, 0xBB04A2C1),
    (0xCB49E3C6, 0xC5626167),
)
_CS64_FACTOR = 0x82893933
_CS64_ADJUSTMENT0 = 0xF2F90B1A
_CS64_ADJUSTMENT1 = 0x0B1A0000

_U32_MASK = 0xFFFFFFFF


def _rol16(value: int) -> int:
    """Rotate a 32-bit value left by 16 bits."""
    value &= _U32_MASK
    return ((value << 16) | (value >> 16)) & _U32_MASK


def _words_le4_even(data: bytes) -> List[int]:
    """Split data into little-endian 32-bit words, zero-padded to an even count."""
    padded = data + b"\x00" * ((-len(data)) % 4)
    if not padded:
        return []
    words = list(struct.unpack(f"<{len(padded) // 4}I", padded))
    if len(words) % 2:
        words.append(0)
    return words


def _sigma0(value: int) -> int:
    """SHA-256 small sigma zero, truncating the combined rotations once."""
    return (
        (value >> 7) ^ (value << 25) ^ (value >> 18) ^ (value << 14) ^ (value >> 3)
    ) & _U32_MASK


def _sigma1(value: int) -> int:
    """SHA-256 small sigma one, truncating the combined rotations once."""
    return (
        (value >> 17) ^ (value << 15) ^ (value >> 19) ^ (value << 13) ^ (value >> 10)
    ) & _U32_MASK


def _sha256_rounds(
    state: Tuple[int, ...], words: List[int], first_round: int = 0
) -> Tuple[int, ...]:
    """Run SHA-256 rounds, optionally resuming inside the first block."""
    a, b, c, d, e, f, g, h = state
    mask = _U32_MASK
    constants = (
        _SHA256_BLOCK_CONSTANTS if first_round == 14 else _SHA256_ROUND_CONSTANTS
    )
    # Multiplying a 32-bit word by 2**32 + 1 repeats it in 64 bits.
    # A right shift then has ROTR's low 32 bits, with no separate left shift.
    # The final additions mask off the unused high bits.
    aligned = min(64, ((first_round + 7) // 8) * 8)
    for index in range(first_round, aligned):
        rot_e = e * 0x100000001
        rot_a = a * 0x100000001
        t1 = (
            h
            + ((rot_e >> 6) ^ (rot_e >> 11) ^ (rot_e >> 25))
            + (g ^ (e & (f ^ g)))
            + constants[index]
            + words[index]
        )
        t2 = ((rot_a >> 2) ^ (rot_a >> 13) ^ (rot_a >> 22)) + ((a & b) | (c & (a | b)))
        a, b, c, d, e, f, g, h = (t1 + t2) & mask, a, b, c, (d + t1) & mask, e, f, g
    # Eight rounds return the working-register names to their original order.
    # Rotate their roles in the expressions instead of moving eight values
    # after every round; only the two updated registers need assignments.
    for index in range(aligned, 64, 8):
        rot_e = e * 0x100000001
        rot_a = a * 0x100000001
        t1 = (
            h
            + ((rot_e >> 6) ^ (rot_e >> 11) ^ (rot_e >> 25))
            + (g ^ (e & (f ^ g)))
            + constants[index]
            + words[index]
        )
        d = (d + t1) & mask
        h = (
            t1
            + ((rot_a >> 2) ^ (rot_a >> 13) ^ (rot_a >> 22))
            + ((a & b) | (c & (a | b)))
        ) & mask
        rot_e = d * 0x100000001
        rot_a = h * 0x100000001
        t1 = (
            g
            + ((rot_e >> 6) ^ (rot_e >> 11) ^ (rot_e >> 25))
            + (f ^ (d & (e ^ f)))
            + constants[index + 1]
            + words[index + 1]
        )
        c = (c + t1) & mask
        g = (
            t1
            + ((rot_a >> 2) ^ (rot_a >> 13) ^ (rot_a >> 22))
            + ((h & a) | (b & (h | a)))
        ) & mask
        rot_e = c * 0x100000001
        rot_a = g * 0x100000001
        t1 = (
            f
            + ((rot_e >> 6) ^ (rot_e >> 11) ^ (rot_e >> 25))
            + (e ^ (c & (d ^ e)))
            + constants[index + 2]
            + words[index + 2]
        )
        b = (b + t1) & mask
        f = (
            t1
            + ((rot_a >> 2) ^ (rot_a >> 13) ^ (rot_a >> 22))
            + ((g & h) | (a & (g | h)))
        ) & mask
        rot_e = b * 0x100000001
        rot_a = f * 0x100000001
        t1 = (
            e
            + ((rot_e >> 6) ^ (rot_e >> 11) ^ (rot_e >> 25))
            + (d ^ (b & (c ^ d)))
            + constants[index + 3]
            + words[index + 3]
        )
        a = (a + t1) & mask
        e = (
            t1
            + ((rot_a >> 2) ^ (rot_a >> 13) ^ (rot_a >> 22))
            + ((f & g) | (h & (f | g)))
        ) & mask
        rot_e = a * 0x100000001
        rot_a = e * 0x100000001
        t1 = (
            d
            + ((rot_e >> 6) ^ (rot_e >> 11) ^ (rot_e >> 25))
            + (c ^ (a & (b ^ c)))
            + constants[index + 4]
            + words[index + 4]
        )
        h = (h + t1) & mask
        d = (
            t1
            + ((rot_a >> 2) ^ (rot_a >> 13) ^ (rot_a >> 22))
            + ((e & f) | (g & (e | f)))
        ) & mask
        rot_e = h * 0x100000001
        rot_a = d * 0x100000001
        t1 = (
            c
            + ((rot_e >> 6) ^ (rot_e >> 11) ^ (rot_e >> 25))
            + (b ^ (h & (a ^ b)))
            + constants[index + 5]
            + words[index + 5]
        )
        g = (g + t1) & mask
        c = (
            t1
            + ((rot_a >> 2) ^ (rot_a >> 13) ^ (rot_a >> 22))
            + ((d & e) | (f & (d | e)))
        ) & mask
        rot_e = g * 0x100000001
        rot_a = c * 0x100000001
        t1 = (
            b
            + ((rot_e >> 6) ^ (rot_e >> 11) ^ (rot_e >> 25))
            + (a ^ (g & (h ^ a)))
            + constants[index + 6]
            + words[index + 6]
        )
        f = (f + t1) & mask
        b = (
            t1
            + ((rot_a >> 2) ^ (rot_a >> 13) ^ (rot_a >> 22))
            + ((c & d) | (e & (c | d)))
        ) & mask
        rot_e = f * 0x100000001
        rot_a = b * 0x100000001
        t1 = (
            a
            + ((rot_e >> 6) ^ (rot_e >> 11) ^ (rot_e >> 25))
            + (h ^ (f & (g ^ h)))
            + constants[index + 7]
            + words[index + 7]
        )
        e = (e + t1) & mask
        a = (
            t1
            + ((rot_a >> 2) ^ (rot_a >> 13) ^ (rot_a >> 22))
            + ((b & c) | (d & (b | c)))
        ) & mask
    return a, b, c, d, e, f, g, h


def _expand_sha256_schedule(words: List[int], first_word: int = 16) -> None:
    """Expand SHA-256 words, combining rotations before modular addition."""
    mask = _U32_MASK
    for index in range(first_word, 64):
        x, y = words[index - 2], words[index - 15]
        # Rotations can retain their high bits until the final sum is masked.
        sigma1 = (x >> 17) ^ (x << 15) ^ (x >> 19) ^ (x << 13) ^ (x >> 10)
        sigma0 = (y >> 7) ^ (y << 25) ^ (y >> 18) ^ (y << 14) ^ (y >> 3)
        words.append((sigma1 + words[index - 7] + sigma0 + words[index - 16]) & mask)


@lru_cache(maxsize=1)
def _sha256_init(data: bytes) -> Tuple[int, ...]:
    """Compute the initial SHA-256 state for token data."""
    mask = _U32_MASK
    words = [0] * 14 + [int.from_bytes(data[:3], "big")]
    words.append(int.from_bytes(data[3:7], "big"))
    for index in range(16, 30):
        value = _SHA256_SCHEDULE[index - 16]
        if index - 2 >= 14:
            value += (
                (_sigma1(words[index - 2]) ^ _SHA256_SIGMA1_ADJUSTMENT)
                if index == 16
                else _sigma1(words[index - 2])
            )
        if index - 7 >= 14:
            value += words[index - 7]
        if index - 15 >= 14:
            value += (
                (_sigma0(words[index - 15]) ^ _SHA256_SIGMA0_ADJUSTMENT)
                if index == 29
                else _sigma0(words[index - 15])
            )
        words.append(value & mask)
    words.append(
        (
            _sigma1(words[28])
            + words[23]
            + _sigma0(words[15])
            + words[14]
            + _SHA256_ADJUSTMENT
            - _SHA256_ROUND_CONSTANTS[14]
        )
        & mask
    )
    _expand_sha256_schedule(words, 31)
    working = _sha256_rounds(_SHA256_STATE, words, 14)
    return tuple((x + y) & mask for x, y in zip(_SHA256_INITIAL, working))


def _sha256_digest(data: bytes) -> bytes:
    """Compute the SHA-256 digest for token data."""
    if len(data) < 7:
        raise ValueError("Token salt must contain at least seven bytes")
    mask = _U32_MASK
    length = 57 + len(data)
    padded = data[7:] + b"\x80" + b"\x00" * ((55 - length) % 64)
    padded += struct.pack(">Q", length * 8)
    state = _sha256_init(data[:7])
    for offset in range(0, len(padded), 64):
        words = list(struct.unpack(">16I", padded[offset : offset + 64]))
        _expand_sha256_schedule(words)
        working = _sha256_rounds(state, words)
        state = tuple((x + y) & mask for x, y in zip(state, working))
    return struct.pack(">8I", *state)


def _cs64_digest(data: bytes, digest: bytes) -> Tuple[int, int]:
    """Compute the CS64 digest words for token data."""
    mask = _U32_MASK
    key_a, key_b = struct.unpack_from("<II", digest)
    key_a |= 1
    key_b |= 1
    factors = (
        (key_a * _CS64_FACTOR) & mask,
        (key_b * _CS64_FACTOR) & mask,
    )
    terms = [
        (coefficient * factors[index % 2]) & mask
        for index, (coefficient, _) in enumerate(_CS64_COEFFICIENTS)
    ]
    offsets = [offset for _, offset in _CS64_COEFFICIENTS]
    for index, word in enumerate(_words_le4_even(b"\x00" + data)):
        term = word * (key_a if index % 2 == 0 else key_b)
        offset = _rol16(word) * (0xEF0569FB if index % 2 == 0 else 0xC31713DB)
        if index == 0:
            term += _CS64_ADJUSTMENT0 * factors[0]
            offset += _CS64_ADJUSTMENT1
        terms.append(term & mask)
        offsets.append(offset & mask)
    swap_acc = swap_last = state = acc = 0
    for index in range(0, len(terms), 2):
        x, y = terms[index], terms[index + 1]
        t = _rol16((offsets[index] + x) & mask)
        q = (t * 0x76F50F34) & mask
        swap_acc += (_rol16(q) * 0xC3EFEA97 + q) & mask
        t = _rol16((offsets[index + 1] + y) & mask)
        q = (t * 0x1881CE3C) & mask
        swap_last = (_rol16(q) * 0x35BD1EC9 + q) & mask
        swap_acc = (swap_acc + swap_last) & mask

        state = (state * key_a + x) & mask
        state = (_rol16(state) * 0xCF98B111) & mask
        state = (_rol16(state) * 0x87085B9F) & mask
        state = (_rol16(state) * 0x12CEB96D) & mask
        state = (_rol16(state) * 0x257E1D83) & mask
        acc += state
        t = (state * key_b + y) & mask
        t = (_rol16(t) * 0xA27416F5) & mask
        t = (_rol16(t) * 0xD38396FF) & mask
        state = (_rol16((_rol16(t) * 0x7C932B89) & mask) * 0xBFA49F69) & mask
        acc = (acc + state) & mask
    return swap_last ^ state, swap_acc ^ acc


@lru_cache(maxsize=1)
def _token_hash(data: bytes) -> Tuple[str, str]:
    """Compute the token hash components."""
    digest = _sha256_digest(data)
    cs64 = struct.pack("<II", *_cs64_digest(data, digest)).hex().upper()
    return cs64, digest.hex().upper()

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
