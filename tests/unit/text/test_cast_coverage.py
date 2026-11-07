"""覆盖率统计（``cast.coverage.json``）：口径、Top-N、音色冲突。"""

from __future__ import annotations

import json

from src.text.cast import AUTO_VOICE, Cast
from src.text.cast_coverage import CoverageReport, classify_speaker

# ------------------------------------------------------------------ 口径


def test_classify_speaker_buckets():
    assert classify_speaker("李明") == "named"
    assert classify_speaker("其他:医生") == "unlisted"
    assert classify_speaker("其他:旁白") == "unlisted"  # 模型笔误也要进长尾清单
    assert classify_speaker("旁白") == "narration"
    assert classify_speaker(None) == "narration"
    assert classify_speaker("?") == "uncertain"
    assert classify_speaker("") == "uncertain"


def test_counts_are_mutually_exclusive_and_sum_to_total():
    r = CoverageReport()
    r.add_chapter(["李明", "其他:医生", "旁白", "?", None, "王芳"])
    assert r.named_quotes == 2
    assert r.unlisted_quotes == 1
    assert r.narration == 2
    assert r.uncertain == 1
    assert r.total == 6
    assert r.chapters == 1


def test_ratio_and_top_coverage():
    r = CoverageReport()
    # 3 个命名台词 + 1 个表外 → 覆盖率 75%，Top1 = 2/4
    r.extend(["李明", "李明", "王芳", "其他:医生"])
    assert r.ratio == 0.75
    assert r.top_coverage(1) == 0.5
    assert r.top_coverage(2) == 0.75
    assert r.top_coverage(99) == 0.75


def test_empty_report_is_zero_not_division_error():
    r = CoverageReport()
    assert r.total == 0
    assert r.ratio == 0.0
    assert r.top_coverage(5) == 0.0
    assert r.to_dict()["quotes"]["total"] == 0


# ------------------------------------------------------------------ 导出


def _cast_with_pool():
    return Cast.from_dict(
        {
            "narrator": "VN",
            "pool": ["V1", "V2"],
            "roles": {"李明": "VLM", "王芳": AUTO_VOICE},
        }
    )


def test_to_dict_structure(tmp_path):
    r = CoverageReport()
    r.add_chapter(["李明", "王芳", "其他:医生", "其他:护士", "旁白", "?"])
    data = r.to_dict(_cast_with_pool(), top=5, unlisted_top=5)

    assert data["quotes"] == {
        "total": 6,
        "named": 2,
        "unlisted": 2,
        "narration": 1,
        "uncertain": 1,
    }
    assert data["coverage"]["overall"] == 0.3333
    assert {x["name"] for x in data["roles"]} == {"李明", "王芳"}
    assert {x["name"] for x in data["unlisted"]} == {"其他:医生", "其他:护士"}
    assert data["cast"]["pool"] == 2 and data["cast"]["auto_roles"] == 1


def test_write_is_valid_json(tmp_path):
    r = CoverageReport()
    r.add_chapter(["李明", "其他:医生", "旁白"])
    target = r.write(tmp_path / "sub" / "cast.coverage.json", _cast_with_pool())
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["quotes"]["total"] == 3
    assert not target.with_suffix(".json.tmp").exists()


# ------------------------------------------------------------------ 音色冲突


def test_voice_conflicts_reports_shared_voice():
    # 池只有 1 个音色 → 所有表外角色必然共用，必须被报出来
    cast = Cast.from_dict({"pool": ["V1"], "roles": {"李明": "VLM"}})
    r = CoverageReport()
    r.extend(["其他:医生", "其他:护士", "李明"])
    conflicts = r.voice_conflicts(cast)
    assert conflicts
    assert conflicts[0]["voice"] == "V1"
    assert set(conflicts[0]["names"]) == {"其他:医生", "其他:护士"}
    assert conflicts[0]["count"] == 2


def test_voice_conflicts_empty_when_all_distinct():
    # 只有一个表外角色 → 没有任何音色被两个名字共用
    cast = Cast.from_dict({"pool": ["V1", "V2"], "roles": {}})
    single = CoverageReport()
    single.extend(["其他:独自一人"])
    assert single.voice_conflicts(cast) == []


def test_voice_conflicts_without_cast_returns_empty():
    r = CoverageReport()
    r.extend(["其他:医生", "其他:护士"])
    assert r.voice_conflicts(None) == []
