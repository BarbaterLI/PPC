"""AudioMerger fail-fast 合并失败单元测试。"""

from pathlib import Path
from unittest.mock import MagicMock, patch

from src.executors.merger import AudioMerger


class _FakeSegment:
    """模拟 pydub AudioSegment。"""
    def __init__(self, duration: int = 1000):
        self._duration = duration
    def __len__(self):
        return self._duration
    def __add__(self, other):
        return self
    def __iadd__(self, other):
        return self
    def export(self, *args, **kwargs):
        pass


def test_merge_fails_when_one_segment_undecodable(tmp_path: Path):
    """任一段解码失败（_load_audio 返回 None）时，合并立即返回失败。"""
    files = [
        tmp_path / "seg_001.mp3",
        tmp_path / "seg_002.mp3",
        tmp_path / "seg_003.mp3",
    ]
    for f in files:
        f.write_bytes(b"ID3")
    output = tmp_path / "merged.mp3"

    fake = _FakeSegment()
    call_count = {"n": 0}

    def fake_load(path):
        call_count["n"] += 1
        if call_count["n"] == 2:
            return None  # 第二个文件解码失败
        return fake

    merger = AudioMerger(silence_ms=0, bitrate="48k")
    with (
        patch.object(merger, "_load_audio", side_effect=fake_load),
        patch("src.executors.merger.AudioSegment.silent", return_value=fake),
    ):
        result = merger.merge(files, output, normalize=False)

    assert result.success is False
    assert result.failed_files is not None and len(result.failed_files) == 1
    assert "无法解码" in result.failed_files[0][1]
    assert not output.exists(), "失败时不应生成输出文件"


def test_merge_fails_when_load_raises_exception(tmp_path: Path):
    """_load_audio 抛异常时，合并立即返回失败。"""
    files = [tmp_path / "seg_001.mp3", tmp_path / "seg_002.mp3"]
    for f in files:
        f.write_bytes(b"ID3")
    output = tmp_path / "merged.mp3"

    fake = _FakeSegment()
    call_count = {"n": 0}

    def fake_load(path):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RuntimeError("ffmpeg error")
        return fake

    merger = AudioMerger(silence_ms=0, bitrate="48k")
    with (
        patch.object(merger, "_load_audio", side_effect=fake_load),
        patch("src.executors.merger.AudioSegment.silent", return_value=fake),
    ):
        result = merger.merge(files, output, normalize=False)

    assert result.success is False
    assert result.failed_files is not None and len(result.failed_files) == 1
    assert "ffmpeg error" in result.failed_files[0][1]


def test_merge_single_corrupt_file_returns_failure(tmp_path: Path):
    """单文件快捷路径：源文件损坏时返回失败而非复制坏文件。"""
    files = [tmp_path / "only.mp3"]
    files[0].write_bytes(b"ID3")
    output = tmp_path / "out.mp3"

    merger = AudioMerger(silence_ms=0, bitrate="48k")
    with patch.object(merger, "_load_audio", return_value=None):
        result = merger.merge(files, output)

    assert result.success is False
    assert result.failed_files is not None and len(result.failed_files) == 1
    assert not output.exists(), "损坏源文件不应被复制到输出"


def test_merge_all_success_file_count_matches(tmp_path: Path):
    """全部段成功合并时 file_count 等于输入文件数。"""
    files = [tmp_path / f"seg_{i:03d}.mp3" for i in range(1, 4)]
    for f in files:
        f.write_bytes(b"ID3")
    output = tmp_path / "merged.mp3"

    fake = _FakeSegment()
    export_mock = MagicMock()

    class _ExportableSegment(_FakeSegment):
        def export(self, *args, **kwargs):
            export_mock(*args, **kwargs)

    exportable = _ExportableSegment()

    merger = AudioMerger(silence_ms=0, bitrate="48k")
    with (
        patch.object(merger, "_load_audio", return_value=exportable),
        patch("src.executors.merger.AudioSegment.silent", return_value=exportable),
    ):
        result = merger.merge(files, output, normalize=False)

    assert result.success is True
    assert result.file_count == 3
