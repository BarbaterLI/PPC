"""_resolve_load_order 依赖拓扑排序单元测试 (Task 9.4)。

覆盖场景：
- 正常依赖链（A 依赖 B，B 应先加载）
- 多层依赖链（A → B → C，顺序应为 C, B, A）
- 循环依赖（A ↔ B，两者均被跳过，记录 WARNING）
- 自依赖（A 依赖 A，A 被跳过，记录 WARNING）
- 缺失依赖（A 依赖不存在的 C，A 被跳过，记录 WARNING）
- 无依赖（保持原顺序）
- 混合场景（部分有依赖、部分无依赖、部分循环）

测试直接调用 ``ExtensionLoader._resolve_load_order`` 方法，
扩展信息字典结构与 ``_discover_extensions`` 返回值保持一致：
``{"type": "file"|"package", "path": Path}``。
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
import yaml

from src.extensions.loader import ExtensionLoader

# ---------------------------------------------------------------------------
# 包构造辅助
# ---------------------------------------------------------------------------


def _make_package(
    tmp_path: Path,
    name: str,
    *,
    dependencies: list[str] | None = None,
) -> Path:
    """创建测试用扩展包目录（含 manifest.yml 与 __init__.py）。

    参数:
        tmp_path: pytest 的临时目录 fixture
        name: 扩展名称（同时作为目录名）
        dependencies: manifest 的 dependencies 字段（依赖的扩展名列表）

    返回:
        包目录路径
    """
    pkg_dir = tmp_path / name
    pkg_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "name": name,
        "version": "1.0.0",
        "package_type": "python",
        "entry_point": "extension:TestExtension",
        "dependencies": dependencies or [],
    }
    (pkg_dir / "manifest.yml").write_text(
        yaml.dump(manifest, allow_unicode=True), encoding="utf-8"
    )
    # __init__.py 使其成为 Python 包目录（_discover_extensions 依赖此文件）
    (pkg_dir / "__init__.py").write_text("", encoding="utf-8")
    return pkg_dir


def _make_file_ext(tmp_path: Path, name: str) -> Path:
    """创建测试用 .py 文件扩展（无 manifest，无依赖）。"""
    file_path = tmp_path / f"{name}.py"
    file_path.write_text(f'"""{name} test extension."""\n', encoding="utf-8")
    return file_path


def _ext_info_package(path: Path) -> dict:
    """构造 package 类型的扩展信息字典。"""
    return {"type": "package", "path": path}


def _ext_info_file(path: Path) -> dict:
    """构造 file 类型的扩展信息字典。"""
    return {"type": "file", "path": path}


def _names_of(ext_infos: list[dict]) -> list[str]:
    """从扩展信息字典列表中提取名称（按目录名或文件 stem）。"""
    names = []
    for info in ext_infos:
        path = info["path"]
        if info.get("type") == "package":
            names.append(path.name)
        else:
            names.append(path.stem)
    return names


# ---------------------------------------------------------------------------
# 测试用例：基础场景
# ---------------------------------------------------------------------------


def test_empty_extensions_returns_empty():
    """空列表输入返回空列表。"""
    loader = ExtensionLoader()
    assert loader._resolve_load_order([]) == []


def test_no_dependencies_preserves_original_order(tmp_path: Path):
    """无依赖关系的扩展保持原始发现顺序。"""
    pkg_a = _make_package(tmp_path, "ext-a")
    pkg_b = _make_package(tmp_path, "ext-b")
    pkg_c = _make_package(tmp_path, "ext-c")

    extensions = [
        _ext_info_package(pkg_a),
        _ext_info_package(pkg_b),
        _ext_info_package(pkg_c),
    ]

    loader = ExtensionLoader()
    result = loader._resolve_load_order(extensions)
    assert _names_of(result) == ["ext-a", "ext-b", "ext-c"]


def test_file_extensions_have_no_dependencies(tmp_path: Path):
    """file 类型扩展无 manifest，视为无依赖，按原顺序返回。"""
    file_a = _make_file_ext(tmp_path, "file_a")
    file_b = _make_file_ext(tmp_path, "file_b")

    extensions = [
        _ext_info_file(file_a),
        _ext_info_file(file_b),
    ]

    loader = ExtensionLoader()
    result = loader._resolve_load_order(extensions)
    assert _names_of(result) == ["file_a", "file_b"]


# ---------------------------------------------------------------------------
# 测试用例：正常依赖链
# ---------------------------------------------------------------------------


def test_simple_dependency_chain(tmp_path: Path):
    """A 依赖 B，B 应先于 A 加载。"""
    pkg_b = _make_package(tmp_path, "ext-b")
    pkg_a = _make_package(tmp_path, "ext-a", dependencies=["ext-b"])

    # 输入顺序为 A, B
    extensions = [
        _ext_info_package(pkg_a),
        _ext_info_package(pkg_b),
    ]

    loader = ExtensionLoader()
    result = loader._resolve_load_order(extensions)
    # 期望 B 在 A 之前
    names = _names_of(result)
    assert names == ["ext-b", "ext-a"]


def test_multi_level_dependency_chain(tmp_path: Path):
    """A → B → C 多层依赖：加载顺序应为 C, B, A。"""
    pkg_c = _make_package(tmp_path, "ext-c")
    pkg_b = _make_package(tmp_path, "ext-b", dependencies=["ext-c"])
    pkg_a = _make_package(tmp_path, "ext-a", dependencies=["ext-b"])

    extensions = [
        _ext_info_package(pkg_a),
        _ext_info_package(pkg_b),
        _ext_info_package(pkg_c),
    ]

    loader = ExtensionLoader()
    result = loader._resolve_load_order(extensions)
    names = _names_of(result)
    assert names == ["ext-c", "ext-b", "ext-a"]


def test_mixed_dependencies_and_independents(tmp_path: Path):
    """混合场景：A→B、C 独立、D→B。B 应先于 A 和 D，C 保持原相对位置。"""
    pkg_b = _make_package(tmp_path, "ext-b")
    pkg_a = _make_package(tmp_path, "ext-a", dependencies=["ext-b"])
    pkg_c = _make_package(tmp_path, "ext-c")
    pkg_d = _make_package(tmp_path, "ext-d", dependencies=["ext-b"])

    # 输入顺序：A, B, C, D
    extensions = [
        _ext_info_package(pkg_a),
        _ext_info_package(pkg_b),
        _ext_info_package(pkg_c),
        _ext_info_package(pkg_d),
    ]

    loader = ExtensionLoader()
    result = loader._resolve_load_order(extensions)
    names = _names_of(result)
    # B 必须先于 A 和 D
    assert names.index("ext-b") < names.index("ext-a")
    assert names.index("ext-b") < names.index("ext-d")
    # 全部 4 个扩展都被加载
    assert sorted(names) == ["ext-a", "ext-b", "ext-c", "ext-d"]


def test_stable_order_for_independent_extensions(tmp_path: Path):
    """多个独立扩展保持原始发现顺序（稳定性验证）。"""
    pkgs = [_make_package(tmp_path, f"ext-{i}") for i in range(5)]
    extensions = [_ext_info_package(p) for p in pkgs]

    loader = ExtensionLoader()
    result = loader._resolve_load_order(extensions)
    assert _names_of(result) == [f"ext-{i}" for i in range(5)]


# ---------------------------------------------------------------------------
# 测试用例：缺失依赖
# ---------------------------------------------------------------------------


def test_missing_dependency_skipped(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    """A 依赖不存在的 C，A 被跳过，记录 WARNING；B 正常加载。"""
    pkg_a = _make_package(tmp_path, "ext-a", dependencies=["ext-c-missing"])
    pkg_b = _make_package(tmp_path, "ext-b")

    extensions = [
        _ext_info_package(pkg_a),
        _ext_info_package(pkg_b),
    ]

    loader = ExtensionLoader()
    with caplog.at_level(logging.WARNING, logger="src.extensions.loader"):
        result = loader._resolve_load_order(extensions)

    # A 被跳过，只剩 B
    names = _names_of(result)
    assert names == ["ext-b"]
    # 记录 WARNING
    assert any(
        "missing dependencies" in rec.message and "ext-a" in rec.message
        for rec in caplog.records
    ), (
        f"expected WARNING about missing dependencies, "
        f"got: {[r.message for r in caplog.records]}"
    )


def test_missing_dependency_does_not_skip_others(tmp_path: Path):
    """A 依赖缺失的 C 被跳过；B、D 独立且正常加载。"""
    pkg_a = _make_package(tmp_path, "ext-a", dependencies=["ext-c-missing"])
    pkg_b = _make_package(tmp_path, "ext-b")
    pkg_d = _make_package(tmp_path, "ext-d")

    extensions = [
        _ext_info_package(pkg_a),
        _ext_info_package(pkg_b),
        _ext_info_package(pkg_d),
    ]

    loader = ExtensionLoader()
    result = loader._resolve_load_order(extensions)
    names = _names_of(result)
    # A 被跳过，B 和 D 按原顺序保留
    assert names == ["ext-b", "ext-d"]


def test_dependency_on_skipped_extension_still_loaded(tmp_path: Path):
    """B 依赖 A，A 因自身缺失依赖被跳过；B 的依赖 A 在已发现集合中，
    B 不属于"缺失依赖"，仍被加载（运行时再处理 A 缺失问题）。
    """
    pkg_a = _make_package(tmp_path, "ext-a", dependencies=["ext-c-missing"])
    pkg_b = _make_package(tmp_path, "ext-b", dependencies=["ext-a"])

    extensions = [
        _ext_info_package(pkg_a),
        _ext_info_package(pkg_b),
    ]

    loader = ExtensionLoader()
    result = loader._resolve_load_order(extensions)
    # A 被跳过（依赖缺失），B 保留（其依赖 A 在已发现集合中）
    names = _names_of(result)
    assert names == ["ext-b"]


# ---------------------------------------------------------------------------
# 测试用例：循环依赖
# ---------------------------------------------------------------------------


def test_circular_dependency_both_skipped(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    """A 依赖 B、B 依赖 A：两者都被跳过，记录 WARNING。"""
    pkg_a = _make_package(tmp_path, "ext-a", dependencies=["ext-b"])
    pkg_b = _make_package(tmp_path, "ext-b", dependencies=["ext-a"])

    extensions = [
        _ext_info_package(pkg_a),
        _ext_info_package(pkg_b),
    ]

    loader = ExtensionLoader()
    with caplog.at_level(logging.WARNING, logger="src.extensions.loader"):
        result = loader._resolve_load_order(extensions)

    # 两者都被跳过
    assert result == []
    # 记录循环依赖 WARNING
    assert any(
        "Circular dependency" in rec.message
        for rec in caplog.records
    ), (
        f"expected WARNING about circular dependency, "
        f"got: {[r.message for r in caplog.records]}"
    )


def test_self_dependency_skipped(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    """A 依赖 A 自身：A 被跳过，记录 WARNING。"""
    pkg_a = _make_package(tmp_path, "ext-a", dependencies=["ext-a"])

    extensions = [_ext_info_package(pkg_a)]

    loader = ExtensionLoader()
    with caplog.at_level(logging.WARNING, logger="src.extensions.loader"):
        result = loader._resolve_load_order(extensions)

    # A 被跳过
    assert result == []
    # 记录循环依赖 WARNING
    assert any(
        "Circular dependency" in rec.message
        for rec in caplog.records
    ), (
        f"expected WARNING about circular dependency, "
        f"got: {[r.message for r in caplog.records]}"
    )


def test_partial_cycle_only_cyclic_skipped(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    """部分循环：A↔B 循环，C 独立。A、B 被跳过，C 正常加载。"""
    pkg_a = _make_package(tmp_path, "ext-a", dependencies=["ext-b"])
    pkg_b = _make_package(tmp_path, "ext-b", dependencies=["ext-a"])
    pkg_c = _make_package(tmp_path, "ext-c")

    extensions = [
        _ext_info_package(pkg_a),
        _ext_info_package(pkg_b),
        _ext_info_package(pkg_c),
    ]

    loader = ExtensionLoader()
    with caplog.at_level(logging.WARNING, logger="src.extensions.loader"):
        result = loader._resolve_load_order(extensions)

    names = _names_of(result)
    # 只有 C 被加载
    assert names == ["ext-c"]
    # 记录循环依赖 WARNING
    assert any(
        "Circular dependency" in rec.message
        for rec in caplog.records
    )


def test_three_node_cycle_all_skipped(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    """三节点循环 A→B→C→A：三者都被跳过，记录 WARNING。"""
    pkg_a = _make_package(tmp_path, "ext-a", dependencies=["ext-b"])
    pkg_b = _make_package(tmp_path, "ext-b", dependencies=["ext-c"])
    pkg_c = _make_package(tmp_path, "ext-c", dependencies=["ext-a"])

    extensions = [
        _ext_info_package(pkg_a),
        _ext_info_package(pkg_b),
        _ext_info_package(pkg_c),
    ]

    loader = ExtensionLoader()
    with caplog.at_level(logging.WARNING, logger="src.extensions.loader"):
        result = loader._resolve_load_order(extensions)

    # 三者都被跳过
    assert result == []
    assert any(
        "Circular dependency" in rec.message
        for rec in caplog.records
    )


# ---------------------------------------------------------------------------
# 测试用例：混合场景
# ---------------------------------------------------------------------------


def test_complex_mixed_scenario(tmp_path: Path):
    """复杂混合场景：A→B→C 正常链 + D↔E 循环 + F 独立 + G→H 缺失依赖。

    期望：
    - C, B, A 按拓扑顺序加载
    - D, E 因循环被跳过
    - F 正常加载
    - G 因依赖 H 缺失被跳过
    """
    pkg_c = _make_package(tmp_path, "ext-c")
    pkg_b = _make_package(tmp_path, "ext-b", dependencies=["ext-c"])
    pkg_a = _make_package(tmp_path, "ext-a", dependencies=["ext-b"])
    pkg_d = _make_package(tmp_path, "ext-d", dependencies=["ext-e"])
    pkg_e = _make_package(tmp_path, "ext-e", dependencies=["ext-d"])
    pkg_f = _make_package(tmp_path, "ext-f")
    pkg_g = _make_package(tmp_path, "ext-g", dependencies=["ext-h-missing"])

    extensions = [
        _ext_info_package(pkg_a),
        _ext_info_package(pkg_b),
        _ext_info_package(pkg_c),
        _ext_info_package(pkg_d),
        _ext_info_package(pkg_e),
        _ext_info_package(pkg_f),
        _ext_info_package(pkg_g),
    ]

    loader = ExtensionLoader()
    result = loader._resolve_load_order(extensions)
    names = _names_of(result)

    # 加载的扩展：A, B, C, F（D, E, G 被跳过）
    assert sorted(names) == ["ext-a", "ext-b", "ext-c", "ext-f"]
    # 拓扑顺序：C 在 B 之前，B 在 A 之前
    assert names.index("ext-c") < names.index("ext-b")
    assert names.index("ext-b") < names.index("ext-a")


def test_no_exception_on_all_circular(tmp_path: Path):
    """全部扩展构成循环时不抛异常，返回空列表。"""
    pkg_a = _make_package(tmp_path, "ext-a", dependencies=["ext-b"])
    pkg_b = _make_package(tmp_path, "ext-b", dependencies=["ext-c"])
    pkg_c = _make_package(tmp_path, "ext-c", dependencies=["ext-a"])

    extensions = [
        _ext_info_package(pkg_a),
        _ext_info_package(pkg_b),
        _ext_info_package(pkg_c),
    ]

    loader = ExtensionLoader()
    # 不应抛异常
    result = loader._resolve_load_order(extensions)
    assert result == []
