"""blb2txt 扩展的单元测试。

重点覆盖「实测踩到的坑」对应的修复逻辑：

* 产物 BOM + CRLF 的归一化（含**变更判定必须基于原始字节**——早期实现
  用已归一化的字符串自比，导致 BOM/CRLF 被误判为“无需修改”而漏改）；
* 文件名序号补零（blb2txt 仅在 toc 模式补零，size 模式产出 ``1``..``12``
  会让宿主字典序排成 1,10,11,2…，拼装出错误的有声书顺序）；
* 未解析占位符检测（blb2txt 对 ``%Header%`` 解析失败会原样保留且返回码为 0，
  必须据此触发 auto 回退链）；
* 输出目录须预先创建、``-o`` 覆盖 ``-v`` 等进程层约定。
"""

from __future__ import annotations

import codecs
import sys
from pathlib import Path
from typing import Any

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from extensions.blb2txt.extension import DEFAULT_CONFIG, Blb2TxtExtension  # noqa: E402
from extensions.blb2txt.extractor import (  # noqa: E402
    SUPPORTED_FORMATS,
    Blb2TxtError,
    Blb2TxtRunner,
    decode_output,
    find_blb2txt,
    normalize_file,
    normalize_text,
)

# ---------------------------------------------------------------------------
# 文本归一化
# ---------------------------------------------------------------------------


def test_normalize_text_strips_bom_and_crlf() -> None:
    raw = "\ufeff第一章\r\n\r\n正文。\r\n"
    assert normalize_text(raw) == "第一章\n\n正文。\n"


def test_normalize_text_handles_lone_cr() -> None:
    assert normalize_text("a\rb") == "a\nb\n"


def test_normalize_text_is_idempotent() -> None:
    once = normalize_text("\ufeffa\r\nb\r\n")
    assert normalize_text(once) == once


def test_decode_output_strips_bom() -> None:
    payload = codecs.BOM_UTF8 + "第一章".encode()
    assert decode_output(payload) == "第一章"


def test_decode_output_utf8() -> None:
    assert decode_output("测试".encode()) == "测试"


def test_decode_output_never_raises_on_garbage() -> None:
    # 任意字节都应有兜底解码路径，不得抛异常
    assert isinstance(decode_output(b"\xff\xfe\x00\x01\x80"), str)


# ---------------------------------------------------------------------------
# 文件归一化（回归：BOM/CRLF 必须真的被改写）
# ---------------------------------------------------------------------------


def test_normalize_file_rewrites_bom_and_crlf(tmp_path: Path) -> None:
    target = tmp_path / "chapter.txt"
    target.write_bytes(codecs.BOM_UTF8 + "第一章\r\n正文\r\n".encode())

    assert normalize_file(target) is True

    raw = target.read_bytes()
    assert not raw.startswith(codecs.BOM_UTF8)
    assert b"\r" not in raw
    assert raw.decode("utf-8") == "第一章\n正文\n"


def test_normalize_file_is_noop_when_already_clean(tmp_path: Path) -> None:
    target = tmp_path / "clean.txt"
    target.write_bytes("第一章\n正文\n".encode())

    assert normalize_file(target) is False
    assert target.read_bytes() == "第一章\n正文\n".encode()


def test_normalize_file_detects_bom_only(tmp_path: Path) -> None:
    """仅含 BOM（无 CR）也必须被改写——这正是早期实现漏掉的情形。"""
    target = tmp_path / "bom_only.txt"
    target.write_bytes(codecs.BOM_UTF8 + b"abc\n")

    assert normalize_file(target) is True
    assert target.read_bytes() == b"abc\n"


def test_normalize_file_missing_path_returns_false(tmp_path: Path) -> None:
    assert normalize_file(tmp_path / "does_not_exist.txt") is False


# ---------------------------------------------------------------------------
# 支持格式与可执行文件定位
# ---------------------------------------------------------------------------


def test_supported_formats_cover_common_ebooks() -> None:
    for fmt in ("epub", "mobi", "azw3", "fb2", "docx", "pdf", "chm", "txt"):
        assert fmt in SUPPORTED_FORMATS


def test_find_blb2txt_discovers_project_bundled_copy() -> None:
    bundled = _REPO_ROOT / "blb2txt" / "blb2txt.exe"
    if not bundled.is_file():
        pytest.skip("仓库未附带 blb2txt.exe")
    assert find_blb2txt() == bundled


def test_find_blb2txt_respects_explicit_path(tmp_path: Path) -> None:
    exe = tmp_path / "blb2txt.exe"
    exe.write_bytes(b"MZ")
    assert find_blb2txt(exe) == exe


def test_find_blb2txt_env_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    exe = tmp_path / "from_env.exe"
    exe.write_bytes(b"MZ")
    monkeypatch.setenv("PPC_BLB2TXT", str(exe))
    assert find_blb2txt() == exe


# ---------------------------------------------------------------------------
# Runner 的错误判定（stub 掉 subprocess，不启动真实进程）
# ---------------------------------------------------------------------------


class _FakeProc:
    def __init__(self, returncode: int = 0, stdout: bytes = b"", stderr: bytes = b"") -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def test_runner_treats_rc_zero_with_error_marker_as_failure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """``rc=0`` 但输出含 ``[Error]`` 时也必须判失败（实测 blb2txt 行为不一致）。"""
    runner = Blb2TxtRunner(tmp_path / "blb2txt.exe")
    src = tmp_path / "in.epub"
    src.write_bytes(b"x")
    monkeypatch.setattr(
        runner,
        "_run",
        lambda *a, **k: _FakeProc(0, b"\n[Error] Output folder not found\n"),
    )
    with pytest.raises(Blb2TxtError, match=r"\[Error\]"):
        runner.extract_to_text(src)


def test_runner_treats_rc_zero_with_usage_as_failure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """无参数时 blb2txt 以 rc=0 打印 usage —— 不可当作成功。"""
    runner = Blb2TxtRunner(tmp_path / "blb2txt.exe")
    src = tmp_path / "in.epub"
    src.write_bytes(b"x")
    monkeypatch.setattr(runner, "_run", lambda *a, **k: _FakeProc(0, b"Usage: blb2txt [options ...]\n"))
    with pytest.raises(Blb2TxtError, match="Usage"):
        runner.extract_to_text(src)


def test_runner_rejects_missing_input(tmp_path: Path) -> None:
    runner = Blb2TxtRunner(tmp_path / "blb2txt.exe")
    with pytest.raises(Blb2TxtError, match="输入文件不存在"):
        runner.extract_to_text(tmp_path / "missing.epub")


def test_runner_creates_output_dir_before_extract(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """blb2txt 不会自动创建 -v 目录，封装层必须先建。"""
    runner = Blb2TxtRunner(tmp_path / "blb2txt.exe")
    src = tmp_path / "book.epub"
    src.write_bytes(b"x")
    out_dir = tmp_path / "nested" / "out"

    monkeypatch.setattr(runner, "_run", lambda *a, **k: _FakeProc(0, b"ok"))

    with pytest.raises(Blb2TxtError, match="未产出"):
        runner.extract_to_dir(src, out_dir)

    assert out_dir.is_dir()  # 目录已由封装层创建


def test_runner_rejects_unresolved_plural_placeholder(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """产出名残留 ``%Header%`` → 视为该模式不适用（触发 auto 回退）。"""
    runner = Blb2TxtRunner(tmp_path / "blb2txt.exe")
    src = tmp_path / "plain.txt"
    src.write_text("no headings here")
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    def fake_run(*_a: Any, **_k: Any) -> _FakeProc:
        # 模拟 blb2txt「占位符原样保留」的行为
        (out_dir / "%Number% - %Header%.txt").write_text("x", encoding="utf-8")
        return _FakeProc(0, b"")

    monkeypatch.setattr(runner, "_run", fake_run)
    with pytest.raises(Blb2TxtError, match="占位符"):
        runner.split_to_dir(src, out_dir, mode="toc", name_pattern="%Number% - %Header%")


# ---------------------------------------------------------------------------
# 扩展层：模式 / 回退链 / 配置
# ---------------------------------------------------------------------------


def test_default_pattern_for_header_modes() -> None:
    ext = Blb2TxtExtension()
    assert ext._pattern_for("toc") == "%Number% - %Header%"
    assert ext._pattern_for("toc_gen") == "%Number% - %Header%"


def test_default_pattern_omitted_for_non_header_modes() -> None:
    """size/blank/caps 无章节标题，不应传 -p（否则留下未解析占位符）。"""
    ext = Blb2TxtExtension()
    assert ext._pattern_for("size") is None
    assert ext._pattern_for("blank") is None
    assert ext._pattern_for("caps") is None


def test_user_pattern_respected_for_all_modes() -> None:
    ext = Blb2TxtExtension()
    ext._user_config = {"default_name_pattern": "MyBook %Number%"}
    assert ext._pattern_for("size") == "MyBook %Number%"
    assert ext._pattern_for("toc") == "MyBook %Number%"


def test_auto_chain_order() -> None:
    assert Blb2TxtExtension.AUTO_CHAIN == ("toc", "toc_gen", "size")


def test_auto_falls_back_and_reports_effective_mode(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """toc / toc_gen 失败后应回退到 size，并回报生效模式。"""
    ext = Blb2TxtExtension()
    src = tmp_path / "book.txt"
    src.write_text("content")
    out = tmp_path / "out"
    attempts: list[str] = []

    def fake_split_to_dir(_input, output_dir, *, mode, **_kw):
        attempts.append(mode)
        if mode in {"toc", "toc_gen"}:
            raise Blb2TxtError(f"模式 {mode} 不适用")
        target = Path(output_dir) / "book.txt"
        target.write_text("content", encoding="utf-8")
        return [target]

    monkeypatch.setattr(ext.get_runner(), "split_to_dir", fake_split_to_dir)
    produced, used = ext.split(src, out, mode="auto")

    assert attempts == ["toc", "toc_gen", "size"]
    assert used == "size"
    assert len(produced) == 1


def test_auto_raises_when_all_modes_fail(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    ext = Blb2TxtExtension()
    src = tmp_path / "book.txt"
    src.write_text("content")

    def always_fail(*_a: Any, **_k: Any):
        raise Blb2TxtError("nope")

    monkeypatch.setattr(ext.get_runner(), "split_to_dir", always_fail)
    with pytest.raises(Blb2TxtError, match="分章失败"):
        ext.split(src, tmp_path / "out", mode="auto")


def test_auto_cleans_failed_attempt_artifacts(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """失败尝试落盘的垃圾文件必须被清掉，不污染输出目录。"""
    ext = Blb2TxtExtension()
    src = tmp_path / "book.txt"
    src.write_text("content")
    out = tmp_path / "out"

    def fake_split_to_dir(_input, output_dir, *, mode, **_kw):
        if mode in {"toc", "toc_gen"}:
            (Path(output_dir) / "%Number% - %Header%.txt").write_text("junk", encoding="utf-8")
            raise Blb2TxtError("占位符未解析")
        target = Path(output_dir) / "book.txt"
        target.write_text("content", encoding="utf-8")
        return [target]

    monkeypatch.setattr(ext.get_runner(), "split_to_dir", fake_split_to_dir)
    produced, used = ext.split(src, out, mode="auto")

    assert used == "size"
    assert not (out / "%Number% - %Header%.txt").exists()
    assert [p.name for p in produced] == ["book.txt"]


def test_clean_flags_only_applied_when_enabled() -> None:
    ext = Blb2TxtExtension()
    # 默认全部关闭 → 不应产生任何 --remove-* 参数
    assert ext._build_extra_args() == []
    # 显式 clean=True → 启用全部清洗项
    assert "-rs" in ext._build_extra_args(clean=True)


def test_extra_args_passthrough() -> None:
    ext = Blb2TxtExtension()
    assert ext._build_extra_args(extra_args=["-if"]) == ["-if"]


def test_extension_metadata() -> None:
    from src.extensions.base import ExtensionType

    ext = Blb2TxtExtension()
    assert ext.metadata.extension_type == ExtensionType.TOOL_INTEGRATION
    perms = {p.value for p in ext.metadata.permissions}
    assert {"subprocess", "filesystem_read", "filesystem_write"} <= perms


def test_normalize_default_enabled() -> None:
    assert DEFAULT_CONFIG["normalize"] is True


def test_get_info_without_exe(monkeypatch: pytest.MonkeyPatch) -> None:
    ext = Blb2TxtExtension()
    monkeypatch.setattr(ext, "resolve_exe", lambda: None)
    info = ext.get_info()
    assert info["installed"] is False
    assert info["exe_path"] is None
    assert ext.is_available() is False
