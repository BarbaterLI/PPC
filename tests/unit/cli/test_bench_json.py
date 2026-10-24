"""``ppc bench --json`` 的 stdout 必须是纯 JSON（P2-14）。

旧实现用全局 ``console.print`` 输出 "▶ bench …" 与 "基线已保存"，
会混进 stdout，导致 ``json.loads(output)`` 失败。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_bench_json_stdout_is_pure_json(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    import src.cli.commands.bench as bench

    monkeypatch.setattr(
        bench,
        "bench_startup",
        lambda iterations: {"wall_ms_p50": 1.0, "wall_ms_min": 1.0, "wall_ms_max": 1.0,
                            "modules_imported": 1, "import_self_ms": 1.0},
    )

    save_path = tmp_path / "baseline.json"
    bench.handle_bench("startup", json_output=True, save=save_path)

    captured = capsys.readouterr()
    # stdout 必须可被 json.loads 直接解析（无人类可读前缀/后缀）
    payload = json.loads(captured.out)
    assert payload["startup"]["wall_ms_p50"] == 1.0
    # 人类可读进度与落盘提示改走 stderr
    assert "bench startup" in captured.err
    assert "基线已保存" in captured.err
    assert "bench startup" not in captured.out
    assert save_path.exists()
