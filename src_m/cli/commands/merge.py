"""音频合并命令实现

支持合并多个音频文件，支持通配符、不同格式混合输入。
"""

import glob
import sys
from pathlib import Path
from typing import List

import typer
from rich.console import Console

console = Console()

merge_app = typer.Typer(help="音频合并工具")


def expand_glob_patterns(patterns: List[str]) -> List[Path]:
    """展开通配符模式"""
    files = []
    for pattern in patterns:
        if '*' in pattern or '?' in pattern:
            matches = glob.glob(pattern)
            for match in matches:
                path = Path(match)
                if path.is_file():
                    files.append(path)
        else:
            path = Path(pattern)
            if path.exists():
                files.append(path)
    return files


@merge_app.command("merge")
def handle_merge(
    input_files: List[str] = typer.Argument(..., help="输入音频文件（支持通配符，如 *.mp3）"),
    output: Path = typer.Option(..., "--output", "-o", help="输出文件路径"),
    silence: int = typer.Option(500, "--silence", "-s", help="音频片段间的静音间隔（毫秒）"),
    format: str = typer.Option("mp3", "--format", "-f", help="输出格式：mp3、wav、ogg"),
    normalize: bool = typer.Option(True, "--normalize/--no-normalize", help="是否归一化音量"),
):
    """合并多个音频文件

    示例:
        ppc9 merge file1.mp3 file2.mp3 -o output.mp3
        ppc9 merge *.mp3 -o all.mp3 --silence 500
        ppc9 merge audio/ -o combined.mp3 --format mp3
    """
    from src_m.executors.merger import AudioMerger

    if not input_files:
        console.print("[red]错误：必须提供至少一个音频文件[/red]")
        raise typer.Exit(1)

    audio_files = expand_glob_patterns(input_files)

    if not audio_files:
        console.print(f"[red]错误：未找到匹配的文件: {', '.join(input_files)}[/red]")
        raise typer.Exit(1)

    valid_extensions = ['.mp3', '.wav', '.ogg', '.m4a', '.aac', '.flac']
    audio_files = [f for f in audio_files if f.suffix.lower() in valid_extensions]
    
    if not audio_files:
        console.print("[red]错误：没有找到有效的音频文件[/red]")
        raise typer.Exit(1)

    audio_files = sorted(audio_files)

    console.print(f"[cyan]将合并 {len(audio_files)} 个音频文件[/cyan]")
    console.print(f"[cyan]静音间隔: {silence}ms[/cyan]")
    console.print(f"[cyan]输出格式: {format}[/cyan]")

    output_path = Path(output)
    if not output_path.suffix:
        output_path = output_path.with_suffix(f".{format}")

    merger = AudioMerger(silence_ms=silence)

    try:
        result = merger.merge(audio_files, output_path, silence_ms=silence, normalize=normalize)

        if result.success:
            console.print(f"\n[green]✓ 合并成功![/green]")
            console.print(f"输出文件: {result.output_path}")
            console.print(f"文件数量: {result.file_count}")
            console.print(f"总时长: {result.duration_seconds:.2f}s")
        else:
            console.print(f"\n[red]✗ 合并失败: {result.error}[/red]")
            raise typer.Exit(1)
    except Exception as e:
        console.print(f"\n[red]✗ 合并失败: {e}[/red]")
        raise typer.Exit(1)


if __name__ == "__main__":
    merge_app()
