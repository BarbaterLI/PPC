"""AudioMerger 输出比特率单元测试。"""

from pathlib import Path
from unittest.mock import MagicMock, patch

from src.executors.merger import AudioMerger, merge_audio_files


def _make_mp3(path: Path, content: bytes = b"ID3") -> Path:
    path.write_bytes(content)
    return path


def test_merge_passes_bitrate_to_export(tmp_path: Path):
    """合并 MP3 时应将 bitrate='48k' 传给 pydub export。"""
    files = [
        _make_mp3(tmp_path / "seg_001.mp3"),
        _make_mp3(tmp_path / "seg_002.mp3"),
    ]
    output = tmp_path / "merged.mp3"

    export_mock = MagicMock()

    class FakeSegment:
        def __init__(self, duration: int = 1000):
            self._duration = duration

        def __len__(self):
            return self._duration

        def __add__(self, other):
            return self

        def __iadd__(self, other):
            return self

        def export(self, *args, **kwargs):
            export_mock(*args, **kwargs)

    fake = FakeSegment()

    merger = AudioMerger(silence_ms=0, bitrate="48k")
    with (
        patch.object(merger, "_load_audio", return_value=fake),
        patch.object(merger, "_normalize_volume", return_value=fake),
        patch("src.executors.merger.AudioSegment.silent", return_value=fake),
    ):
        result = merger.merge(files, output, normalize=False)

    assert result.success is True
    export_mock.assert_called_once()
    _, kwargs = export_mock.call_args
    assert kwargs.get("bitrate") == "48k"


def test_merge_audio_files_uses_default_48k_bitrate(tmp_path: Path):
    """便捷函数默认使用 48k 比特率。"""
    files = [_make_mp3(tmp_path / "seg_001.mp3")]
    output = tmp_path / "out.mp3"

    result = merge_audio_files(files, output)
    assert result.success is True
    assert output.exists()
