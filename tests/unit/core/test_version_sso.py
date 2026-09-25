"""版本号单一来源守护测试。

约定：当前应用版本号字符串只允许出现在 ``src/version.py``（及用户数据
``config.yml``、测试夹具）中；``src/`` 其余源码与配置模板必须引用
``src.version.__version__`` 或使用 ``{{version}}`` 占位符，避免升版时漂移。
"""

from __future__ import annotations

from pathlib import Path

from src.version import __version__

SRC_ROOT = Path(__file__).resolve().parents[3] / "src"


def _iter_source_files():
    for path in sorted(SRC_ROOT.rglob("*.py")):
        yield path
    yield SRC_ROOT / "config" / "templates" / "config_template.yml"


def test_current_version_literal_only_in_version_py():
    offenders: list[str] = []
    for path in _iter_source_files():
        text = path.read_text(encoding="utf-8")
        if __version__ in text and path.name != "version.py":
            offenders.append(path.relative_to(SRC_ROOT.parent).as_posix())
    assert not offenders, f"版本号 {__version__} 字面量散落在: {offenders}（应引用 src.version）"


def test_version_is_valid_semver():
    parts = __version__.split(".")
    assert len(parts) == 3, f"版本号应为 x.y.z: {__version__}"
    assert parts[0].isdigit() and parts[1].isdigit()
    tail = parts[2]
    digits = ""
    for ch in tail:
        if ch.isdigit():
            digits += ch
        else:
            break
    assert digits, f"补丁号缺失: {__version__}"


def test_pyproject_version_matches():
    import tomllib

    pyproject = SRC_ROOT.parent / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    assert data["project"]["version"] == __version__
