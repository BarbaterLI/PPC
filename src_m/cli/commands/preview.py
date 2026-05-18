"""音频预览命令实现

支持预览 TTS 音频片段。
"""

import asyncio
import os
import tempfile
from pathlib import Path

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()

preview_app = typer.Typer(help="音频预览工具")


@preview_app.command("preview")
def handle_preview(
    text: str = typer.Argument(..., help="要预览的文本内容"),
    voice: str = typer.Option("zh-CN-XiaoxiaoNeural", "--voice", "-v", help="语音名称"),
    rate: str = typer.Option("+0%", "--rate", "-r", help="语速调整（如 +10%, -5%）"),
    output: Path = typer.Option(None, "--output", "-o", help="输出文件路径（可选）"),
    duration: int = typer.Option(10, "--duration", "-d", help="预览最大时长（秒）"),
):
    """预览 TTS 音频片段

    示例:
        ppc9 preview "这是一段预览文本"
        ppc9 preview "快速语音" --voice zh-CN-YunxiNeural --rate +20%
        ppc9 preview "保存到文件" -o preview.mp3
    """
    import edge_tts
    import uuid

    async def generate_preview():
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("正在生成预览...", total=None)

            if output is None:
                temp_dir = tempfile.gettempdir()
                output = Path(temp_dir) / f"ppc9_preview_{uuid.uuid4().hex[:8]}.mp3"

            output_path = Path(output)
            progress.update(task, description=f"正在合成音频: {output_path.name}")

            try:
                communicate = edge_tts.Communicate(text, voice, rate=rate)
                await communicate.save(str(output_path))

                progress.update(task, description="完成")
                console.print(f"\n[green]✓ 预览生成成功![/green]")
                console.print(f"文件: {output_path}")
                console.print(f"语音: {voice}")
                console.print(f"语速: {rate}")
                
                return True
            except Exception as e:
                console.print(f"\n[red]✗ 生成失败: {e}[/red]")
                return False

    success = asyncio.run(generate_preview())
    if not success:
        raise typer.Exit(1)


@preview_app.command("voices")
def list_voices(
    language: str = typer.Option(None, "--language", "-l", help="按语言筛选（如 zh-CN）"),
):
    """列出可用的语音

    示例:
        ppc9 preview voices
        ppc9 preview voices --language zh-CN
    """
    import edge_tts

    async def get_voices():
        try:
            voices = await edge_tts.list_voices()

            if language:
                voices = [v for v in voices if v.get("Locale", "").startswith(language)]

            console.print(f"\n[cyan]可用语音 ({len(voices)} 个):[/cyan]\n")

            current_locale = None
            for voice in voices:
                locale = voice.get("Locale", "")
                name = voice.get("ShortName", "")
                gender = voice.get("Gender", "")
                friendly_name = voice.get("FriendlyName", "")

                if locale != current_locale:
                    console.print(f"\n[bold]{locale}[/bold]")
                    current_locale = locale

                console.print(f"  {name} ({gender}) - {friendly_name}")

        except Exception as e:
            console.print(f"[red]获取语音列表失败: {e}[/red]")
            raise typer.Exit(1)

    asyncio.run(get_voices())


if __name__ == "__main__":
    preview_app()
