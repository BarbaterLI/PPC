"""语音列表命令 - 列出所有可用 TTS 语音。

注：原有的系统检查 (SystemChecker / handle_check) 已整合到
`analyze` 命令中（默认模式为轻量级健康检查，--deep 启用深度分析）。
"""

import asyncio
from typing import Any

from src.cli.typer_app import get_output

from ..errors import CLIError
from ..errors import ErrorCode as E


async def _list_voices() -> list[Any]:
    """获取可用语音列表。"""
    try:
        import edge_tts

        voices: list[Any] = await edge_tts.list_voices()  # type: ignore[no-any-return]  # edge_tts 无 stub
        return voices
    except Exception:
        return []


def handle_voices(json_output: bool = False):
    """处理语音列表命令 - 列出所有可用语音。"""
    output = get_output()
    if json_output:
        output.set_mode(json_output=True)

    try:
        voices = asyncio.run(_list_voices())

        if not voices:
            raise CLIError(
                E.E_NETWORK,
                "无法获取语音列表(edge_tts 列表为空或网络不通)",
                hint="检查网络连接或代理设置;稍后重试",
            )

        records: list[dict[str, Any]] = []
        for voice in voices:
            records.append(
                {
                    "name": voice.get("ShortName", ""),
                    "locale": voice.get("Locale", ""),
                    "gender": voice.get("Gender", ""),
                    "description": voice.get("FriendlyName", ""),
                }
            )

        def _is_zh(r):
            return str(r.get("locale", "")).startswith("zh-")

        records_sorted = sorted(records, key=lambda r: (not _is_zh(r), r.get("name", "")))

        headers = ["Name", "Locale", "Gender", "Description"]
        rows = [
            [r.get("name", ""), r.get("locale", ""), r.get("gender", ""), r.get("description", "")]
            for r in records_sorted
        ]

        if output.mode == "json":
            output.print_table(headers, rows, title=None, json_data=records_sorted)
            return

        output.show_banner()
        output.print_table(headers, rows, title=f"PPC10 可用语音列表 (共 {len(records)} 个)")
        output.info("提示：使用 'ppc10 config set --key tts.voice --value <语音 ID>' 设置默认语音")
        output.info("示例：ppc10 config set --key tts.voice --value zh-CN-XiaoxiaoNeural")

    except CLIError:
        raise
    except Exception as e:
        raise CLIError(
            E.E_NETWORK,
            f"获取语音列表失败: {e}",
            hint="检查网络连接或稍后重试",
        ) from e
