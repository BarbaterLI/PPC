"""语音列表命令 - 列出所有可用 TTS 语音。

注：原有的系统检查 (SystemChecker / handle_check) 已整合到
`analyze` 命令中（默认模式为轻量级健康检查，--deep 启用深度分析）。
"""

import asyncio
import sys
from typing import Any, Dict, List

from rich.box import SIMPLE
from rich.table import Table

from ..output import OutputFormatter, BrandColors, Icons
from ..errors import CLIError, ErrorCode as E


async def _list_voices() -> List[Dict[str, Any]]:
    """获取可用语音列表。"""
    try:
        import edge_tts
        voices = await edge_tts.list_voices()
        return voices
    except Exception:
        return []


def handle_voices(json_output: bool = False):
    """处理语音列表命令 - 列出所有可用语音。"""
    output = OutputFormatter(verbose=False)
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

        # 构造行 + dict 数组
        records: List[Dict[str, Any]] = []
        for voice in voices:
            records.append({
                "name": voice.get("ShortName", ""),
                "locale": voice.get("Locale", ""),
                "gender": voice.get("Gender", ""),
                "description": voice.get("FriendlyName", ""),
            })

        # 中文优先排序(仅影响人类显示顺序;JSON 模式保持原始顺序)
        def _is_zh(r):
            return str(r.get("locale", "")).startswith("zh-")

        records_sorted = sorted(records, key=lambda r: (not _is_zh(r), r.get("name", "")))

        headers = ["Name", "Locale", "Gender", "Description"]
        rows = [
            [r.get("name", ""), r.get("locale", ""), r.get("gender", ""), r.get("description", "")]
            for r in records_sorted
        ]

        if output.mode == "json":
            # 直接 dump 数组(由 print_table 装配 dict 列表)
            output.print_table(headers, rows, title=None, json_data=records_sorted)
            return

        # human 模式
        output.show_banner()
        output.console.print(
            f"\n[bold {BrandColors.PRIMARY}]{'═' * 60}[/bold {BrandColors.PRIMARY}]"
        )
        output.console.print(
            f"[bold white]  {Icons.MICROPHONE} PPC10 可用语音列表[/bold white]"
        )
        output.console.print(
            f"[bold {BrandColors.PRIMARY}]{'═' * 60}[/bold {BrandColors.PRIMARY}]\n"
        )
        output.console.print(
            f"[bold {BrandColors.SUCCESS}]找到 {len(records)} 个可用语音[/bold {BrandColors.SUCCESS}]\n"
        )
        output.print_table(headers, rows, title=None)
        output.console.print(
            f"\n[dim]提示：使用 'ppc10 config set --key tts.voice --value <语音 ID>' 设置默认语音[/dim]"
        )
        output.console.print(
            f"[dim]示例：ppc10 config set --key tts.voice --value zh-CN-XiaoxiaoNeural[/dim]\n"
        )

    except CLIError:
        raise
    except Exception as e:
        raise CLIError(
            E.E_NETWORK,
            f"获取语音列表失败: {e}",
            hint="检查网络连接或稍后重试",
        ) from e
