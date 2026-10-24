"""balcon TTS 后端扩展的单元测试。

覆盖参数换算（语速/音量/音调/采样率）、语音列表解析、协议面与可用性判定。

这些换算的正确性来自对 ``balcon.exe`` 的实测标定：语速满足
``speed_factor = 3 ** (s / 10)``（在 s=-10..10 的 11 个采样点上与实测
最大偏差 < 1%），故此处以公式反解值作为断言基准。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from extensions.balcon.extension import BACKEND_NAME, DEFAULT_CONFIG, BalconExtension  # noqa: E402
from extensions.balcon.ffmpeg_tool import (  # noqa: E402
    DOWNLOAD_HINT,
    PCM_FORMAT_BY_DEPTH,
    check_ffmpeg,
    encode_pcm_to_mp3,
    find_ffmpeg,
)
from extensions.balcon.runner import (  # noqa: E402
    SUPPORTED_KHZ,
    BalconRunner,
    find_balcon,
    hz_to_pitch_steps,
    khz_to_hz,
    nearest_supported_khz,
    parse_voices,
    percent_to_sapi_rate,
    percent_to_volume,
)

# ---------------------------------------------------------------------------
# 语速换算（实测标定：speed = 3 ** (s / 10)）
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("rate", "expected"),
    [
        ("+0%", 0),
        ("0%", 0),
        ("", 0),
        ("+40%", 3),      # 10*log3(1.4) = 3.06
        ("+100%", 6),     # 10*log3(2.0) = 6.31
        ("-50%", -6),     # 10*log3(0.5) = -6.31
        ("+200%", 10),    # 10*log3(3.0) = 10.0
        ("-90%", -10),    # 越界应被夹取
        ("+500%", 10),
        ("abc", 0),       # 无法解析回退原速
    ],
)
def test_percent_to_sapi_rate(rate: str, expected: int) -> None:
    assert percent_to_sapi_rate(rate) == expected


def test_percent_to_sapi_rate_monotonic() -> None:
    """语速档位应随百分比单调不减。"""
    values = [percent_to_sapi_rate(f"{pct:+d}%") for pct in range(-90, 210, 10)]
    assert values == sorted(values)
    assert all(-10 <= v <= 10 for v in values)


# ---------------------------------------------------------------------------
# 音量 / 音调换算
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("volume", "expected"),
    [
        ("+0%", 100),
        ("", 100),
        ("-30%", 70),
        ("+50%", 100),   # 夹取到满量
        ("-150%", 0),    # 夹取到静音
    ],
)
def test_percent_to_volume(volume: str, expected: int) -> None:
    assert percent_to_volume(volume) == expected


@pytest.mark.parametrize(
    ("pitch", "expected"),
    [
        ("+0Hz", 0),
        ("", 0),
        ("+50Hz", 5),
        ("-50Hz", -5),
        ("-100Hz", -10),
        ("+200Hz", 10),  # 夹取
    ],
)
def test_hz_to_pitch_steps(pitch: str, expected: int) -> None:
    assert hz_to_pitch_steps(pitch) == expected


# ---------------------------------------------------------------------------
# 采样率归到 balcon -fr 支持值
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("hz", "expected_khz"),
    [
        (8000, 8),
        (11025, 11),
        # 12 档不可用（balcon -fr 12 输出 0 字节），故 12000 归到最近的 11
        (12000, 11),
        (16000, 16),
        (22050, 22),
        (24000, 24),
        (32000, 32),
        (44100, 44),
        (48000, 48),
        (6000, 8),      # 低于最小值 → 取最近
        (50000, 48),    # 高于最大值 → 取最近
    ],
)
def test_nearest_supported_khz(hz: int, expected_khz: int) -> None:
    assert nearest_supported_khz(hz) == expected_khz
    assert nearest_supported_khz(hz) in SUPPORTED_KHZ


def test_default_sample_rate_is_edge_compatible() -> None:
    """默认参数应与 Edge TTS 的 24kHz 单声道 48kbps 对齐。"""
    assert DEFAULT_CONFIG["sample_rate"] == 24000
    assert DEFAULT_CONFIG["channels"] == 1
    assert DEFAULT_CONFIG["bitrate"] == "48k"


# ---------------------------------------------------------------------------
# balcon -l 输出解析
# ---------------------------------------------------------------------------


def test_parse_voices_with_sections() -> None:
    output = "\r\nSAPI 5:\r\n  Microsoft Huihui Desktop\r\n  Microsoft Zira Desktop\r\n"
    voices = parse_voices(output)
    assert [v["name"] for v in voices] == [
        "Microsoft Huihui Desktop",
        "Microsoft Zira Desktop",
    ]
    assert all(v["engine"] == "SAPI 5" for v in voices)


def test_parse_voices_multiple_sections() -> None:
    output = "\nSAPI 5:\n  Voice A\nSAPI 4:\n  Voice B\n"
    voices = parse_voices(output)
    assert voices[0] == {"name": "Voice A", "engine": "SAPI 5"}
    assert voices[1] == {"name": "Voice B", "engine": "SAPI 4"}


def test_parse_voices_empty() -> None:
    assert parse_voices("") == []
    assert parse_voices("\n\n") == []


def test_parse_voices_ignores_headers_without_colon() -> None:
    output = "\nsome header line\n  Real Voice\n"
    assert [v["name"] for v in parse_voices(output)] == ["Real Voice"]


# ---------------------------------------------------------------------------
# 可执行文件定位
# ---------------------------------------------------------------------------


def test_find_balcon_respects_explicit_path(tmp_path: Path) -> None:
    exe = tmp_path / "balcon.exe"
    exe.write_bytes(b"MZ")
    assert find_balcon(exe) == exe


def test_find_balcon_falls_back_when_explicit_path_missing(tmp_path: Path) -> None:
    """显式路径不存在时应继续回退探测，而不是直接失败。

    断言回退结果**确实指向一个存在的候选**（而非仅仅"不等于缺失路径"——
    后者恒真，属无效断言）。
    """
    missing = tmp_path / "nope.exe"
    resolved = find_balcon(missing)
    bundled = _REPO_ROOT / "balcon" / "balcon.exe"
    if bundled.is_file():
        assert resolved == bundled
    else:
        assert resolved is None or resolved.is_file()


def test_find_balcon_env_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    exe = tmp_path / "from_env.exe"
    exe.write_bytes(b"MZ")
    monkeypatch.setenv("PPC_BALCON", str(exe))
    assert find_balcon() == exe


def test_find_balcon_discovers_project_bundled_copy() -> None:
    """仓库内 ``<root>/balcon/balcon.exe`` 应被自动发现。"""
    bundled = _REPO_ROOT / "balcon" / "balcon.exe"
    if not bundled.is_file():
        pytest.skip("仓库未附带 balcon.exe")
    assert find_balcon() == bundled


# ---------------------------------------------------------------------------
# ffmpeg 探测与下载引导
# ---------------------------------------------------------------------------


def test_download_hint_mentions_ffmpeg_and_actions() -> None:
    assert "ffmpeg" in DOWNLOAD_HINT.lower()
    # 引导必须给出可执行的下一步，而不是只报错
    assert "winget" in DOWNLOAD_HINT
    assert "ffmpeg_path" in DOWNLOAD_HINT


def test_find_ffmpeg_respects_explicit_path(tmp_path: Path) -> None:
    fake = tmp_path / "ffmpeg.exe"
    fake.write_bytes(b"MZ")
    assert find_ffmpeg(fake) == fake


def test_check_ffmpeg_reports_hint_when_missing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """确定性覆盖「ffmpeg 缺失」分支：打桩 find_ffmpeg 返回 None。

    原实现直接在真实机器上探测，本机装有 ffmpeg 时条件体永不执行（近似空测），
    故此处显式打桩以保证该分支必然被走到。
    """
    from extensions.balcon import ffmpeg_tool as mod

    monkeypatch.setattr(mod, "find_ffmpeg", lambda *a, **k: None)
    ok, detail, path = mod.check_ffmpeg()
    assert ok is False
    assert path is None
    assert "ffmpeg" in detail.lower()
    assert "winget" in detail  # 必须给出可执行的下一步，而非只报错


def test_find_ffmpeg_env_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    exe = tmp_path / "ffmpeg_env.exe"
    exe.write_bytes(b"MZ")
    monkeypatch.setenv("PPC_FFMPEG", str(exe))
    assert find_ffmpeg() == exe


# ---------------------------------------------------------------------------
# 采样率映射（回归：-fr 档位不等于 kHz×1000）
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("khz", "expected_hz"),
    [
        (8, 8000),
        (11, 11025),   # 不是 11000
        (16, 16000),
        (22, 22050),   # 不是 22000
        (24, 24000),
        (32, 32000),
        (44, 44100),   # 不是 44000
        (48, 48000),
    ],
)
def test_khz_to_hz_uses_real_rates(khz: int, expected_hz: int) -> None:
    """``-fr 11/22/44`` 实测为 11025/22050/44100，不能简单乘 1000。"""
    assert khz_to_hz(khz) == expected_hz


def test_supported_khz_excludes_unusable_12() -> None:
    """实测 ``balcon -fr 12`` 返回码 0 但输出 0 字节，必须从支持列表剔除。"""
    assert 12 not in SUPPORTED_KHZ


def test_nearest_supported_khz_never_maps_to_12() -> None:
    for hz in (11000, 11500, 12000, 12500, 13000):
        assert nearest_supported_khz(hz) != 12


def test_runner_output_sample_rate_is_real(tmp_path: Path) -> None:
    runner = BalconRunner(tmp_path / "balcon.exe", sample_rate=22050)
    assert runner.output_khz == 22
    assert runner.output_sample_rate == 22050


def test_runner_default_sample_rate_is_exact(tmp_path: Path) -> None:
    runner = BalconRunner(tmp_path / "balcon.exe", sample_rate=24000)
    assert runner.output_sample_rate == 24000


# ---------------------------------------------------------------------------
# 位深 → ffmpeg PCM 格式（回归：8 位曾被按 s16le 解释）
# ---------------------------------------------------------------------------


def test_pcm_format_mapping_matches_balcon_signedness() -> None:
    """实测 `-bt 8` 输出为**无符号**（均值居中 128），故必须映射到 u8。"""
    assert PCM_FORMAT_BY_DEPTH[8] == "u8"
    assert PCM_FORMAT_BY_DEPTH[16] == "s16le"


def test_encode_rejects_unsupported_bit_depth(tmp_path: Path) -> None:
    """非法位深应在启动 ffmpeg 之前就被拒绝，而不是产出错音频。"""
    with pytest.raises(Exception, match="位深"):
        encode_pcm_to_mp3(
            b"\x00" * 64,
            tmp_path / "out.mp3",
            ffmpeg_path=tmp_path / "whatever",
            sample_rate=24000,
            channels=1,
            bit_depth=24,
        )


def test_encode_pcm_to_mp3_rejects_missing_ffmpeg(tmp_path: Path) -> None:
    with pytest.raises(Exception) as exc:
        encode_pcm_to_mp3(
            b"\x00" * 100,
            tmp_path / "out.mp3",
            ffmpeg_path=tmp_path / "no_such_ffmpeg",
            sample_rate=24000,
            channels=1,
        )
    assert "ffmpeg" in str(exc.value).lower()


# ---------------------------------------------------------------------------
# Runner 参数构造（不启动真实进程）
# ---------------------------------------------------------------------------


def test_runner_synthesize_rejects_empty_text(tmp_path: Path) -> None:
    runner = BalconRunner(tmp_path / "balcon.exe")
    with pytest.raises(Exception, match="文本为空"):
        runner.synthesize_pcm("   ", "Any Voice")


def test_runner_reports_output_khz(tmp_path: Path) -> None:
    runner = BalconRunner(tmp_path / "balcon.exe", sample_rate=22050)
    assert runner.output_khz == 22


# ---------------------------------------------------------------------------
# 扩展协议面
# ---------------------------------------------------------------------------


def test_extension_protocol_surface() -> None:
    ext = BalconExtension()
    assert ext.get_name() == BACKEND_NAME
    assert ext.protocol_version() == "ppc.tts.v1"
    caps = ext.capabilities()
    assert caps["voices"] is True
    assert caps["probe"] is True


def test_extension_declares_subprocess_permission() -> None:
    ext = BalconExtension()
    perms = {p.value for p in ext.metadata.permissions}
    assert "subprocess" in perms
    assert "filesystem_read" in perms
    assert "filesystem_write" in perms


def test_extension_is_tts_backend_type() -> None:
    from src.extensions.base import ExtensionType

    assert BalconExtension().metadata.extension_type == ExtensionType.TTS_BACKEND


def test_synthesize_rejects_non_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    """非 Windows 平台必须在合成入口直接拒绝。"""
    import asyncio
    import importlib

    # 注意：``extensions.balcon.extension`` 包属性指向**实例**（__init__.py 中
    # ``extension = BalconExtension()``），必须用 importlib 取到真实模块。
    mod = importlib.import_module("extensions.balcon.extension")
    monkeypatch.setattr(mod.os, "name", "posix")

    ext = BalconExtension()
    with pytest.raises(Exception, match="Windows"):
        asyncio.run(ext.synthesize("hello", "Any Voice"))


def test_is_available_false_off_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib

    mod = importlib.import_module("extensions.balcon.extension")
    monkeypatch.setattr(mod.os, "name", "posix")
    assert BalconExtension().is_available() is False


def test_user_config_overrides_defaults() -> None:
    """用户运行期配置应覆盖内置默认值。"""
    ext = BalconExtension()
    ext._user_config = {"sample_rate": 16000, "voice": "Test Voice"}
    assert ext.config["sample_rate"] == 16000
    assert ext.config["voice"] == "Test Voice"
    # 未覆盖的键仍取默认值
    assert ext.config["bitrate"] == DEFAULT_CONFIG["bitrate"]


def test_voice_for_priority() -> None:
    """语音优先级：入参 > 扩展配置 > 空（balcon 默认）。"""
    ext = BalconExtension()
    assert ext._voice_for("Explicit") == "Explicit"
    ext._user_config = {"voice": "Configured"}
    assert ext._voice_for("") == "Configured"
    ext._user_config = {}
    assert ext._voice_for("") == ""
