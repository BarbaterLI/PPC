"""Communicate with the service. Only the Communicate class should be used by
end-users. The other classes and functions are for internal use only."""

import asyncio
import json
import ssl
import time
import uuid
from contextlib import nullcontext
from dataclasses import dataclass
from io import TextIOWrapper
from typing import (
    AsyncGenerator,
    ContextManager,
    Dict,
    Generator,
    List,
    Optional,
    Tuple,
    Union,
)
from xml.sax.saxutils import escape, unescape

import aiohttp
import certifi
from typing_extensions import Literal

from .constants import (
    DEFAULT_VOICE,
    MP3_BITRATE_BPS,
    SEC_MS_GEC_VERSION,
    TICKS_PER_SECOND,
    WSS_HEADERS,
    WSS_URL,
)
from .data_classes import TTSConfig
from .drm import DEFAULT_DRM, DRM
from .exceptions import (
    NoAudioReceived,
    UnexpectedResponse,
    UnknownResponse,
    WebSocketError,
)
from .typing import CommunicateState, TTSChunk

_SSL_CTX = ssl.create_default_context(cafile=certifi.where())

# 服务不支持的字符范围（0-8, 11, 12, 14-31）一次性映射为空格。
# 在模块级构建翻译表，避免每次调用重复构建；``str.translate`` 比逐字符
# ``ord()`` 判断 + 列表重建快 5-10 倍，对长文本（如 2500 字符分段）显著降耗。
# 注意：制表符(9)、换行(10)、回车(13) 故意保留，不在替换范围内。
_INCOMPATIBLE_CHARS_TABLE = str.maketrans(
    dict.fromkeys((*range(0, 9), 11, 12, *range(14, 32)), " ")
)


def get_headers_and_data(
    data: bytes, header_length: int
) -> Tuple[Dict[bytes, bytes], bytes]:
    """
    Returns the headers and data from the given data.

    Args:
        data (bytes): The data to be parsed.
        header_length (int): The length of the header.

    Returns:
        tuple: The headers and data to be used in the request.
    """
    if not isinstance(data, bytes):
        raise TypeError("data must be bytes")

    headers = {}
    for line in data[:header_length].split(b"\r\n"):
        key, value = line.split(b":", 1)
        headers[key] = value

    return headers, data[header_length + 2 :]


def remove_incompatible_characters(string: Union[str, bytes]) -> str:
    """
    The service does not support a couple character ranges.
    Most important being the vertical tab character which is
    commonly present in OCR-ed PDFs. Not doing this will
    result in an error from the service.

    Args:
        string (str or bytes): The string to be cleaned.

    Returns:
        str: The cleaned string.
    """
    if isinstance(string, bytes):
        string = string.decode("utf-8")
    if not isinstance(string, str):
        raise TypeError("string must be str or bytes")

    return string.translate(_INCOMPATIBLE_CHARS_TABLE)


def connect_id() -> str:
    """
    Returns a UUID without dashes.

    Returns:
        str: A UUID without dashes.
    """
    return uuid.uuid4().hex


def _find_last_newline_or_space_within_limit(text: bytes, limit: int) -> int:
    """
    Finds the index of the rightmost preferred split character (newline or space)
    within the initial `limit` bytes of the text.

    This helps find a natural word or sentence boundary for splitting, prioritizing
    newlines over spaces.

    Args:
        text (bytes): The byte string to search within.
        limit (int): The maximum index (exclusive) to search up to.

    Returns:
        int: The index of the last found newline or space within the limit,
             or -1 if neither is found in that range.
    """
    # Prioritize finding a newline character
    split_at = text.rfind(b"\n", 0, limit)
    # If no newline is found, search for a space
    if split_at < 0:
        split_at = text.rfind(b" ", 0, limit)
    return split_at


def _find_safe_utf8_split_point(text_segment: bytes) -> int:
    """
    Finds the rightmost possible byte index such that the
    segment `text_segment[:index]` is a valid UTF-8 sequence.

    This prevents splitting in the middle of a multi-byte UTF-8 character.

    Args:
        text_segment (bytes): The byte segment being considered for splitting.

    Returns:
        int: The index of the safe split point. Returns 0 if no valid split
             point is found (e.g., if the first byte is part of a multi-byte
             sequence longer than the limit allows).
    """
    split_at = len(text_segment)
    while split_at > 0:
        try:
            text_segment[:split_at].decode("utf-8")
            # Found the largest valid UTF-8 sequence
            return split_at
        except UnicodeDecodeError:
            # The byte at split_at-1 is part of an incomplete multi-byte char, try earlier
            split_at -= 1

    return split_at


def _adjust_split_point_for_xml_entity(text: bytes, split_at: int) -> int:
    """
    Adjusts a proposed split point backward to prevent splitting inside an XML entity.

    For example, if `text` is `b"this &amp; that"` and `split_at` falls between
    `&` and `;`, this function moves `split_at` to the index before `&`.

    Args:
        text (bytes): The text segment being considered.
        split_at (int): The proposed split point index, determined by whitespace
                        or UTF-8 safety.

    Returns:
        int: The adjusted split point index. It will be moved to the '&'
             if an unterminated entity is detected right before the original `split_at`.
             Otherwise, the original `split_at` is returned.
    """
    while split_at > 0 and b"&" in text[:split_at]:
        ampersand_index = text.rindex(b"&", 0, split_at)
        # Check if a semicolon exists between the ampersand and the split point
        if text.find(b";", ampersand_index, split_at) != -1:
            # Found a terminated entity (like &amp;), safe to break at original split_at
            break

        # Ampersand is not terminated before split_at, move split_at to it
        split_at = ampersand_index

    return split_at


def split_text_by_byte_length(
    text: Union[str, bytes], byte_length: int
) -> Generator[bytes, None, None]:
    """
    Splits text into chunks, each not exceeding a maximum byte length.

    This function prioritizes splitting at natural boundaries (newlines, spaces)
    while ensuring that:
    1. No chunk exceeds `byte_length` bytes.
    2. Chunks do not end with an incomplete UTF-8 multi-byte character.
    3. Chunks do not split XML entities (like `&amp;`) in the middle.

    Args:
        text (str or bytes): The input text. If str, it's encoded to UTF-8.
        byte_length (int): The maximum allowed byte length for any yielded chunk.
                           Must be positive.

    Yields:
        bytes: Text chunks (UTF-8 encoded, stripped of leading/trailing whitespace)
               that conform to the byte length and integrity constraints.

    Raises:
        TypeError: If `text` is not str or bytes.
        ValueError: If `byte_length` is not positive, or if a split point
                    cannot be determined (e.g., due to extremely small byte_length
                    relative to character/entity sizes).
    """
    if isinstance(text, str):
        text = text.encode("utf-8")
    if not isinstance(text, bytes):
        raise TypeError("text must be str or bytes")

    if byte_length <= 0:
        raise ValueError("byte_length must be greater than 0")

    while len(text) > byte_length:
        # Find the initial split point based on whitespace or UTF-8 boundary
        split_at = _find_last_newline_or_space_within_limit(text, byte_length)

        if split_at < 0:
            # 触发条件：前 ``byte_length`` 字节内没有任何空格/换行（中文长段落、
            # 无标点的连续文本很常见）。原实现把**整段剩余文本**传给
            # ``_find_safe_utf8_split_point``，该函数总是返回 ``len(text)``，
            # 于是这里产出一个远超上限的单块（实测 9000B 中文 → 单个 9000B 块），
            # WebSocket 请求体会超限或被服务端拒绝。
            # 修复：只在 ``text[:byte_length]`` 范围内找 UTF-8 字符边界，即
            # 退化为「按字节上限硬切」，同时保证不会截断多字节序列。
            split_at = _find_safe_utf8_split_point(text[:byte_length])

        # Adjust the split point to avoid cutting in the middle of an xml entity, such as '&amp;'
        split_at = _adjust_split_point_for_xml_entity(text, split_at)

        if split_at <= 0:
            # ``byte_length`` 小于单个字符的字节数（如 3 字节汉字配 2 字节上限），
            # 或文本以未闭合的 ``&`` 实体开头导致切分点被回退到 0。
            # 此时无法既满足上限又不产生碎片，直接报错而不是死循环/产出超限块。
            raise ValueError(
                "Maximum byte length is too small or "
                "invalid text structure near '&' or invalid UTF-8"
            )

        # Yield the chunk
        chunk = text[:split_at].strip()
        if chunk:
            yield chunk

        # Prepare for the next iteration
        # If split_at became 0 after adjustment, advance by 1 to avoid infinite loop
        text = text[split_at if split_at > 0 else 1 :]

    # Yield the remaining part
    remaining_chunk = text.strip()
    if remaining_chunk:
        yield remaining_chunk


def mkssml(tc: TTSConfig, escaped_text: Union[str, bytes]) -> str:
    """
    Creates a SSML string from the given parameters.

    Args:
        tc (TTSConfig): The TTS configuration.
        escaped_text (str or bytes): The escaped text. If bytes, it must be UTF-8 encoded.

    Returns:
        str: The SSML string.
    """
    if isinstance(escaped_text, bytes):
        escaped_text = escaped_text.decode("utf-8")

    return (
        "<speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis' xml:lang='en-US'>"
        f"<voice name='{tc.voice}'>"
        f"<prosody pitch='{tc.pitch}' rate='{tc.rate}' volume='{tc.volume}'>"
        f"{escaped_text}"
        "</prosody>"
        "</voice>"
        "</speak>"
    )


def date_to_string() -> str:
    """
    Return Javascript-style date string.

    Returns:
        str: Javascript-style date string.
    """
    # %Z is not what we want, but it's the only way to get the timezone
    # without having to use a library. We'll just use UTC and hope for the best.
    # For example, right now %Z would return EEST when we need it to return
    # Eastern European Summer Time.
    return time.strftime(
        "%a %b %d %Y %H:%M:%S GMT+0000 (Coordinated Universal Time)", time.gmtime()
    )


def ssml_headers_plus_data(request_id: str, timestamp: str, ssml: str) -> str:
    """
    Returns the headers and data to be used in the request.

    Returns:
        str: The headers and data to be used in the request.
    """

    return (
        f"X-RequestId:{request_id}\r\n"
        "Content-Type:application/ssml+xml\r\n"
        f"X-Timestamp:{timestamp}Z\r\n"  # This is not a mistake, Microsoft Edge bug.
        "Path:ssml\r\n\r\n"
        f"{ssml}"
    )


@dataclass
class CommunicateConfig:
    """TTS / transport settings for :class:`Communicate`.

    Collects the former multi-parameter constructor into a single
    validated configuration object.
    """

    voice: str = DEFAULT_VOICE
    rate: str = "+0%"
    volume: str = "+0%"
    pitch: str = "+0Hz"
    boundary: Literal["WordBoundary", "SentenceBoundary"] = "SentenceBoundary"
    connector: Optional[aiohttp.BaseConnector] = None
    proxy: Optional[str] = None
    connect_timeout: Optional[int] = 10
    receive_timeout: Optional[int] = 60

    def __post_init__(self) -> None:
        # Validate the proxy parameter.
        if self.proxy is not None and not isinstance(self.proxy, str):
            raise TypeError("proxy must be str")

        # Validate the timeout parameters.
        if not isinstance(self.connect_timeout, int):
            raise TypeError("connect_timeout must be int")
        if not isinstance(self.receive_timeout, int):
            raise TypeError("receive_timeout must be int")

        # Validate the connector parameter.
        if self.connector is not None and not isinstance(self.connector, aiohttp.BaseConnector):
            raise TypeError("connector must be aiohttp.BaseConnector")

    def make_tts_config(self) -> TTSConfig:
        """Build the :class:`TTSConfig` (validates voice/rate/volume/pitch)."""
        return TTSConfig(self.voice, self.rate, self.volume, self.pitch, self.boundary)

    def make_session_timeout(self) -> aiohttp.ClientTimeout:
        """Build the per-session :class:`aiohttp.ClientTimeout`."""
        return aiohttp.ClientTimeout(
            total=None,
            connect=None,
            sock_connect=self.connect_timeout,
            sock_read=self.receive_timeout,
        )


class Communicate:
    """
    Communicate with the service.
    """

    def __init__(
        self,
        text: str,
        voice: str = DEFAULT_VOICE,
        *,
        rate: str = "+0%",
        volume: str = "+0%",
        pitch: str = "+0Hz",
        boundary: Literal["WordBoundary", "SentenceBoundary"] = "SentenceBoundary",
        connector: Optional[aiohttp.BaseConnector] = None,
        proxy: Optional[str] = None,
        connect_timeout: Optional[int] = 10,
        receive_timeout: Optional[int] = 60,
        drm: Optional[DRM] = None,
    ):
        # Validate the text parameter.
        if not isinstance(text, str):
            raise TypeError("text must be str")

        # Validate TTS/transport settings and store the configuration.
        self.config = CommunicateConfig(
            voice=voice,
            rate=rate,
            volume=volume,
            pitch=pitch,
            boundary=boundary,
            connector=connector,
            proxy=proxy,
            connect_timeout=connect_timeout,
            receive_timeout=receive_timeout,
        )
        # Validate TTS settings and store the TTSConfig object.
        self.tts_config = self.config.make_tts_config()

        # Split the text into multiple strings and store them.
        self.texts = split_text_by_byte_length(
            escape(remove_incompatible_characters(text)),
            4096,
        )

        self.proxy: Optional[str] = self.config.proxy
        self.connector: Optional[aiohttp.BaseConnector] = self.config.connector
        self.session_timeout = self.config.make_session_timeout()

        # DRM instance (clock skew state) — shared default keeps the
        # 403-driven skew correction across all streams in this process.
        self._drm: DRM = drm if drm is not None else DEFAULT_DRM

        # Per-request isolation hooks (set externally by CommunicateAdapter).
        # ``_muid``: when set, used instead of a random MUID on each request —
        # avoids the class-level ``DRM.generate_muid`` mutation that caused
        # race conditions under concurrent streams.
        # ``_extra_headers``: merged on top of ``WSS_HEADERS`` for each
        # request — used for per-stream User-Agent rotation without mutating
        # the shared ``BASE_HEADERS`` dict.
        self._muid: Optional[str] = None
        self._extra_headers: Dict[str, str] = {}

        # Store current state of TTS.
        self.state: CommunicateState = CommunicateState()

    def _parse_metadata(self, data: bytes) -> List[TTSChunk]:
        """解析一条 ``audio.metadata`` 帧，返回其中**全部** boundary 块。

        触发条件：服务端把多个 ``WordBoundary``/``SentenceBoundary`` 事件
        合并进同一条 TEXT 帧下发（协议未承诺单条下发；此处做防御性加固）。
        后果（旧实现）：只解析第一条就 return，后续 boundary 被静默丢弃 →
        逐字高亮/字幕时间轴缺字。遍历全部事件后，若服务端实际只发一条，
        返回长度为 1 的列表，行为与旧实现完全一致（非回归）。

        未知类型仍按旧语义处理：**首个**有效 boundary 之前出现未知类型即抛
        :class:`UnknownResponse`；之后出现则忽略（旧实现此时已 return）。
        """
        chunks: List[TTSChunk] = []
        for meta_obj in json.loads(data)["Metadata"]:
            meta_type = meta_obj["Type"]
            if meta_type in ("WordBoundary", "SentenceBoundary"):
                current_offset = (
                    meta_obj["Data"]["Offset"] + self.state.offset_compensation
                )
                current_duration = meta_obj["Data"]["Duration"]
                chunks.append(
                    {
                        "type": meta_type,
                        "offset": current_offset,
                        "duration": current_duration,
                        "text": unescape(meta_obj["Data"]["text"]["Text"]),
                    }
                )
                continue
            if meta_type in ("SessionEnd",):
                continue
            if not chunks:
                raise UnknownResponse(f"Unknown metadata type: {meta_type}")
        if not chunks:
            raise UnexpectedResponse("No WordBoundary metadata found")
        return chunks

    def _compensate_offset(self) -> None:
        """Update inter-chunk offset_compensation from cumulative CBR audio bytes.

        The output format is audio-24khz-48kbitrate-mono-mp3 (48 kbps CBR).
        For any CBR stream the byte-to-tick conversion is exact integer
        arithmetic:  ticks = total_bytes * 8 * 10_000_000 // 48_000.

        This replaces the previous metadata-based accumulation which drifted
        on long texts due to variable AI silence and Microsoft's integer
        overflow in reported offsets.
        """
        self.state.cumulative_audio_bytes += self.state.chunk_audio_bytes
        self.state.offset_compensation = (
            self.state.cumulative_audio_bytes
            * 8
            * TICKS_PER_SECOND
            // MP3_BITRATE_BPS
        )
        self.state.chunk_audio_bytes = 0

    def _build_request_headers(self) -> Dict[str, str]:
        """Build per-request headers for the WebSocket connection.

        Build per-request headers: start from WSS_HEADERS, apply per-stream
        ``_extra_headers`` (e.g. rotated User-Agent), then inject the MUID
        via ``DRM.headers_with_muid`` using the per-stream ``_muid`` when
        set (avoids the class-level ``DRM.generate_muid`` mutation).
        """
        request_headers = WSS_HEADERS.copy()
        request_headers.update(self._extra_headers)
        return self._drm.headers_with_muid(request_headers, muid=self._muid)

    async def _send_command_request(self, websocket) -> None:
        """Sends the command request to the service."""
        word_boundary = self.tts_config.boundary == "WordBoundary"
        wd = "true" if word_boundary else "false"
        sq = "true" if not word_boundary else "false"
        await websocket.send_str(
            f"X-Timestamp:{date_to_string()}\r\n"
            "Content-Type:application/json; charset=utf-8\r\n"
            "Path:speech.config\r\n\r\n"
            '{"context":{"synthesis":{"audio":{"metadataoptions":{'
            f'"sentenceBoundaryEnabled":"{sq}","wordBoundaryEnabled":"{wd}"'
            "},"
            '"outputFormat":"audio-24khz-48kbitrate-mono-mp3"'
            "}}}}\r\n"
        )

    async def _send_ssml_request(self, websocket) -> None:
        """Sends the SSML request to the service."""
        await websocket.send_str(
            ssml_headers_plus_data(
                connect_id(),
                date_to_string(),
                mkssml(
                    self.tts_config,
                    self.state.partial_text,
                ),
            )
        )

    def _handle_text_message(
        self, received
    ) -> Tuple[Optional[List[TTSChunk]], bool]:
        """Handle a TEXT websocket message.

        Returns ``(chunks, is_turn_end)``: 该帧解析出的全部 metadata 块
        （可能为空 → None），以及本轮流是否结束。
        """
        encoded_data: bytes = received.data.encode("utf-8")
        parameters, data = get_headers_and_data(
            encoded_data, encoded_data.find(b"\r\n\r\n")
        )

        path = parameters.get(b"Path", None)
        if path == b"audio.metadata":
            # Parse the metadata and yield it.
            parsed_metadata = self._parse_metadata(data)

            # Update the last duration offset for use by the next SSML request.
            last = parsed_metadata[-1]
            self.state.last_duration_offset = last["offset"] + last["duration"]
            return parsed_metadata, False
        if path == b"turn.end":
            # Compute inter-chunk offset from actual CBR
            # audio bytes (see _compensate_offset).
            self._compensate_offset()
            return None, True
        if path not in (b"response", b"turn.start"):
            raise UnknownResponse("Unknown path received")
        return None, False

    def _parse_audio_message(self, received) -> Optional[bytes]:
        """Handle a BINARY websocket message; return the audio payload.

        Returns None for the expected empty stream-termination frame;
        raises :class:`UnexpectedResponse` for malformed messages.
        """
        # Message is too short to contain header length.
        if len(received.data) < 2:
            raise UnexpectedResponse(
                "We received a binary message, but it is missing the header length."
            )

        # The first two bytes of the binary message contain the header length.
        header_length = int.from_bytes(received.data[:2], "big")
        if header_length > len(received.data):
            raise UnexpectedResponse(
                "The header length is greater than the length of the data."
            )

        # Parse the headers and data from the binary message.
        parameters, data = get_headers_and_data(
            received.data, header_length
        )

        # Check if the path is audio.
        if parameters.get(b"Path") != b"audio":
            raise UnexpectedResponse(
                "Received binary message, but the path is not audio."
            )

        # At termination of the stream, the service sends a binary message
        # with no Content-Type; this is expected. What is not expected is for
        # an audio stream to be sent with no data.
        content_type = parameters.get(b"Content-Type", None)
        if content_type not in (b"audio/mpeg", None):
            raise UnexpectedResponse(
                "Received binary message, but with an unexpected Content-Type."
            )

        # We only allow no Content-Type if there is no data.
        if content_type is None:
            if len(data) == 0:
                return None

            # If the data is not empty, then we need to raise an exception.
            raise UnexpectedResponse(
                "Received binary message with no Content-Type, but with data."
            )

        # If the data is empty now, then we need to raise an exception.
        if len(data) == 0:
            raise UnexpectedResponse(
                "Received binary message, but it is missing the audio data."
            )

        return data

    async def _stream(self) -> AsyncGenerator[TTSChunk, None]:
        # audio_was_received indicates whether we have received audio data
        # from the websocket. This is so we can raise an exception if we
        # don't receive any audio data.
        audio_was_received = False

        request_headers = self._build_request_headers()

        # Create a new connection to the service.
        async with aiohttp.ClientSession(
            connector=self.connector,
            trust_env=True,
            timeout=self.session_timeout,
        ) as session, session.ws_connect(
            f"{WSS_URL}&ConnectionId={connect_id()}"
            f"&Sec-MS-GEC={self._drm.generate_sec_ms_gec()}"
            f"&Sec-MS-GEC-Version={SEC_MS_GEC_VERSION}",
            compress=15,
            proxy=self.proxy,
            headers=request_headers,
            ssl=_SSL_CTX,
        ) as websocket:
            await self._send_command_request(websocket)

            await self._send_ssml_request(websocket)

            async for received in websocket:
                if received.type == aiohttp.WSMsgType.TEXT:
                    parsed_metadata, is_turn_end = self._handle_text_message(received)
                    if is_turn_end:
                        break
                    if parsed_metadata:
                        # 一帧可能携带多个 boundary 事件：逐个 yield，
                        # 避免只取首个导致后续时间轴缺失。
                        for item in parsed_metadata:
                            yield item
                elif received.type == aiohttp.WSMsgType.BINARY:
                    data = self._parse_audio_message(received)
                    if data is None:
                        continue
                    # Yield the audio data and count bytes for offset compensation.
                    audio_was_received = True
                    self.state.chunk_audio_bytes += len(data)
                    yield {"type": "audio", "data": data}
                elif received.type == aiohttp.WSMsgType.ERROR:
                    raise WebSocketError(
                        received.data if received.data else "Unknown error"
                    )

            if not audio_was_received:
                raise NoAudioReceived(
                    "No audio was received. Please verify that your parameters are correct."
                )

    async def stream(
        self,
    ) -> AsyncGenerator[TTSChunk, None]:
        """
        Streams audio and metadata from the service.

        Raises:
            NoAudioReceived: If no audio is received from the service.
            UnexpectedResponse: If the response from the service is unexpected.
            UnknownResponse: If the response from the service is unknown.
            WebSocketError: If there is an error with the websocket.
        """

        # Check if stream was called before.
        if self.state.stream_was_called:
            raise RuntimeError("stream can only be called once.")
        self.state.stream_was_called = True

        # Stream the audio and metadata from the service.
        for self.state.partial_text in self.texts:
            self.state.chunk_audio_bytes = 0
            try:
                async for message in self._stream():
                    yield message
            except aiohttp.ClientResponseError as e:
                if e.status != 403:
                    raise

                self._drm.handle_client_response_error(e)
                self.state.chunk_audio_bytes = 0
                async for message in self._stream():
                    yield message

    async def save(
        self,
        audio_fname: Union[str, bytes],
        metadata_fname: Optional[Union[str, bytes]] = None,
    ) -> None:
        """
        Save the audio and metadata to the specified files.
        """
        metadata: Union[TextIOWrapper, ContextManager[None]] = (
            open(metadata_fname, "w", encoding="utf-8")
            if metadata_fname is not None
            else nullcontext()
        )
        with metadata, open(audio_fname, "wb") as audio:
            async for message in self.stream():
                if message["type"] == "audio":
                    audio.write(message["data"])
                elif isinstance(metadata, TextIOWrapper) and message["type"] in (
                    "WordBoundary",
                    "SentenceBoundary",
                ):
                    json.dump(message, metadata)
                    metadata.write("\n")
