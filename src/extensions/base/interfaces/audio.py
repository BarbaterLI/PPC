"""音频处理接口（音频转换/TTS 合成后端）。"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class AudioConverter(ABC):
    """音频转换器接口（转换后处理）。

    用户可实现此接口来创建自定义音频转换器扩展，在单个 TTS 小任务
    合成完成后对输出音频做后处理（如调用 ffmpeg 转格式/参数）。
    """

    @abstractmethod
    def is_available(self) -> bool:
        """检查转换器是否可用（依赖/工具是否就绪）。

        返回:
            可用返回 True，否则返回 False（调用方应安全跳过）。
        """
        pass

    @abstractmethod
    def convert(self, input_path: Path | str, output_path: Path | str, options: dict[str, Any] | None = None) -> Path | str | None:
        """转换音频文件。

        参数:
            input_path: 输入音频文件路径。
            output_path: 输出音频文件路径。
            options: 转换选项（目标格式、比特率、采样率等）。

        返回:
            产出文件的路径；转换失败可抛出异常或返回 None（由调用方决定）。
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """返回转换器名称。"""
        pass


class TTSBackend(ABC):
    """TTS 合成后端提供者接口（``ppc.tts.v1`` 协议）。

    必选面是 ``get_name`` / ``is_available`` / ``synthesize``；下面的
    ``protocol_version`` / ``capabilities`` / ``list_voices`` / ``probe``
    为协议 v1 的可选能力面，均带安全默认实现——早期扩展无需改动即可通过
    协商（视为 v1 基线、无可选能力）。

    本类面向扩展作者。宿主内置后端按协议鸭子类型实现、不继承本类，以保持
    ``src.engines`` 不依赖扩展框架（见 ``src.engines.tts_protocol``）。
    """

    @abstractmethod
    def is_available(self) -> bool:
        """检查后端是否可用（依赖/服务是否就绪）。

        返回:
            可用返回 True，否则返回 False。
        """
        pass

    @abstractmethod
    async def synthesize(
        self,
        text: str,
        voice: str,
        *,
        rate: str = "+0%",
        volume: str = "+0%",
        pitch: str = "+0Hz",
        output_path: Path | str | None = None,
        **kwargs: Any,
    ) -> Any:
        """合成一段文本为音频。

        参数:
            text: 待合成的文本。
            voice: 语音名称。
            rate: 语速（如 "+0%"）。
            volume: 音量（如 "+0%"）。
            pitch: 音调（如 "+0Hz"）。
            output_path: 输出音频文件路径；None 时返回字节数据。
            kwargs: 后端相关附加参数。

        返回:
            输出文件路径或音频字节数据；失败抛异常。
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """返回后端名称。"""
        pass

    # ---- ppc.tts.v1 可选能力面（默认实现保证向后兼容） ----------------------

    def protocol_version(self) -> str:
        """本后端遵循的协议版本；默认声明 v1。"""
        return "ppc.tts.v1"

    def capabilities(self) -> dict[str, Any]:
        """能力声明（``voices`` 可枚举语音、``probe`` 可健康探测）。

        默认按子类是否真正覆盖 ``list_voices`` / ``probe`` 如实上报，
        避免基类空实现被宿主误判为具备能力。
        """
        return {
            "voices": type(self).list_voices is not TTSBackend.list_voices,
            "probe": type(self).probe is not TTSBackend.probe,
        }

    async def list_voices(self) -> list[dict[str, Any]]:
        """枚举可用语音（VoiceInfo 兼容 dict 列表）；默认不支持。"""
        return []

    async def probe(self) -> tuple[bool, str]:
        """轻量健康探测，返回 ``(可用, 说明)``；默认等价于 ``is_available``。"""
        return bool(self.is_available()), ""
