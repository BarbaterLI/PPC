"""``ppc doctor`` 自诊断命令测试。"""

from __future__ import annotations

from src.cli.commands import doctor as doc


def test_run_doctor_no_network_produces_full_report():
    rep = doc.run_doctor(use_network=False)
    names = [c.name for c in rep.checks]
    assert "Python 版本" in names
    assert "网络可达性" in names  # 跳过也留痕
    assert len(rep.checks) >= 8
    # 任何单项检查都不允许把整体打崩：状态枚举合法
    assert all(c.status in ("ok", "warn", "fail") for c in rep.checks)
    skip = next(c for c in rep.checks if c.name == "网络可达性")
    assert "跳过" in skip.detail


def test_check_exception_isolated_as_warn(monkeypatch):
    def boom(rep):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(
        doc, "_CHECKS", [("ffmpeg", boom), ("python", doc.check_python), ("ffmpeg2", boom)]
    )
    rep = doc.run_doctor(use_network=False)
    warns = [c for c in rep.checks if c.status == "warn" and "kaboom" in c.detail]
    assert len(warns) == 2  # 两个崩溃项各自被隔离
    # 崩溃项之后仍继续其余检查
    assert any(c.name == "Python 版本" for c in rep.checks)


def test_report_sorting_puts_failures_first():
    rep = doc.Report()
    rep.add("a", "ok")
    rep.add("b", "fail", "x")
    rep.add("c", "warn", "y")
    rep.checks.sort(key=lambda c: (-c.rank, c.name))
    assert [c.name for c in rep.checks] == ["b", "c", "a"]
    assert rep.worst == "fail" and rep.failed


def test_summary_counts_and_exit_code(tmp_path, capsys):
    rep = doc.run_doctor(use_network=False)
    summary = rep.to_dict()["summary"]
    assert summary["ok"] + summary["warn"] + summary["fail"] == len(rep.checks)
    code = doc.handle_doctor(None, use_network=False, json_output=True)
    out = capsys.readouterr().out
    assert out.lstrip().startswith("{")  # 合法 JSON 直通
    assert code in (0, 1)


def test_missing_config_path_is_fail(tmp_path):
    rep = doc.run_doctor(tmp_path / "nope.yml", use_network=False)
    bad = [c for c in rep.checks if c.name == "配置文件"]
    assert bad and bad[0].status == "fail"
