"""ExtensionManifest 新字段（package_type / entry_point / requirements 等）测试。

验证 Task 3 中新增的 7 个字段的解析、默认值与 validate_package 校验规则。
"""

import zipfile
from pathlib import Path

import yaml

from src.extensions.base import ExtensionMetadata, ExtensionType
from src.extensions.manifest import (
    ExtensionManifest,
    _build_manifest,
)
from src.extensions.package import validate_package

# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

EXTENSION_PY_CONTENT = '"""Test extension."""\nprint("hello")\n'

# 64 位十六进制 SHA-256 示例
_VALID_SHA256 = "a" * 64


def _create_zip(
    zip_path: Path,
    manifest: dict | None = None,
    files: dict[str, str] | None = None,
) -> Path:
    """创建测试用扩展包 zip 文件。"""
    files = files or {}
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        if manifest is not None:
            zf.writestr("manifest.yml", yaml.dump(manifest, allow_unicode=True))
        for name, content in files.items():
            zf.writestr(name, content)
    return zip_path


def _valid_manifest(name: str = "manifest-fields-pkg", **overrides) -> dict:
    """生成一个有效的 manifest 字典（含必填字段）。"""
    manifest = {
        "name": name,
        "version": "1.0.0",
        "description": "Test extension package",
        "author": "tester",
        "extension_type": "tool_integration",
        "entry": "extension.py",
        "dependencies": [],
        "min_ppc_version": "11.0.0rc1",
        "tags": [],
        "permissions": [],
    }
    manifest.update(overrides)
    return manifest


# ---------------------------------------------------------------------------
# 默认值与字段解析测试
# ---------------------------------------------------------------------------


def test_default_package_type_python():
    """manifest 未指定 package_type 时默认为 'python'。"""
    raw = _valid_manifest()
    manifest = _build_manifest(raw)

    assert manifest.package_type == "python"
    assert manifest.entry_point == ""
    assert manifest.requirements == []
    assert manifest.config_schema is None
    assert manifest.allowed_hosts == []
    assert manifest.resource_limits == {}
    assert manifest.sha256 == ""


def test_parse_new_fields():
    """manifest 含所有新字段时正确解析。"""
    raw = _valid_manifest(
        package_type="executable",
        entry_point="extension:ExampleExtension",
        requirements=["requests", "pyyaml"],
        config_schema={"type": "object", "properties": {"foo": {"type": "string"}}},
        allowed_hosts=["api.example.com", "cdn.example.com"],
        resource_limits={"max_cpu_percent": 50, "max_memory_mb": 256, "timeout_seconds": 30},
        sha256=_VALID_SHA256,
    )
    manifest = _build_manifest(raw)

    assert manifest.package_type == "executable"
    assert manifest.entry_point == "extension:ExampleExtension"
    assert manifest.requirements == ["requests", "pyyaml"]
    assert manifest.config_schema == {
        "type": "object",
        "properties": {"foo": {"type": "string"}},
    }
    assert manifest.allowed_hosts == ["api.example.com", "cdn.example.com"]
    assert manifest.resource_limits == {
        "max_cpu_percent": 50,
        "max_memory_mb": 256,
        "timeout_seconds": 30,
    }
    assert manifest.sha256 == _VALID_SHA256


# ---------------------------------------------------------------------------
# validate_package 校验测试
# ---------------------------------------------------------------------------


def test_validate_python_entry_point_format(tmp_path: Path):
    """python 类型：有效 entry_point 通过；缺少冒号的 entry_point 报错。"""
    # 有效 entry_point
    zip_ok = tmp_path / "ok.ppcext.zip"
    _create_zip(
        zip_ok,
        manifest=_valid_manifest(
            package_type="python",
            entry_point="extension:MyExtension",
        ),
        files={"extension.py": EXTENSION_PY_CONTENT},
    )
    report = validate_package(zip_ok)
    assert report.is_valid, f"有效 entry_point 应通过校验，错误: {report.messages}"

    # 无效 entry_point（无冒号）
    zip_bad = tmp_path / "bad.ppcext.zip"
    _create_zip(
        zip_bad,
        manifest=_valid_manifest(
            package_type="python",
            entry_point="no_colon_here",
        ),
        files={"extension.py": EXTENSION_PY_CONTENT},
    )
    report = validate_package(zip_bad)
    assert not report.is_valid
    assert any("entry_point" in e and "格式无效" in e for e in report.messages)

    # 无效 entry_point（冒号前为空）
    zip_empty_mod = tmp_path / "empty_mod.ppcext.zip"
    _create_zip(
        zip_empty_mod,
        manifest=_valid_manifest(
            package_type="python",
            entry_point=":MyExtension",
        ),
        files={"extension.py": EXTENSION_PY_CONTENT},
    )
    report = validate_package(zip_empty_mod)
    assert not report.is_valid
    assert any("entry_point" in e and "格式无效" in e for e in report.messages)

    # 无效 entry_point（冒号后为空）
    zip_empty_cls = tmp_path / "empty_cls.ppcext.zip"
    _create_zip(
        zip_empty_cls,
        manifest=_valid_manifest(
            package_type="python",
            entry_point="extension:",
        ),
        files={"extension.py": EXTENSION_PY_CONTENT},
    )
    report = validate_package(zip_empty_cls)
    assert not report.is_valid
    assert any("entry_point" in e and "格式无效" in e for e in report.messages)


def test_validate_executable_requires_sha256(tmp_path: Path):
    """executable 类型：缺少 sha256 失败；提供 64 位十六进制 sha256 通过。"""
    # 缺少 sha256
    zip_no_hash = tmp_path / "no_hash.ppcext.zip"
    _create_zip(
        zip_no_hash,
        manifest=_valid_manifest(
            package_type="executable",
            entry_point="extension:ExeExtension",
            sha256="",
        ),
        files={"extension.py": EXTENSION_PY_CONTENT},
    )
    report = validate_package(zip_no_hash)
    assert not report.is_valid
    assert any("sha256" in e and "必须提供" in e for e in report.messages)

    # 提供 64 位十六进制 sha256 通过
    zip_with_hash = tmp_path / "with_hash.ppcext.zip"
    _create_zip(
        zip_with_hash,
        manifest=_valid_manifest(
            package_type="executable",
            entry_point="extension:ExeExtension",
            sha256=_VALID_SHA256,
        ),
        files={"extension.py": EXTENSION_PY_CONTENT},
    )
    report = validate_package(zip_with_hash)
    assert report.is_valid, f"有效 executable 包应通过校验，错误: {report.messages}"

    # 长度错误的 sha256
    zip_short_hash = tmp_path / "short_hash.ppcext.zip"
    _create_zip(
        zip_short_hash,
        manifest=_valid_manifest(
            package_type="executable",
            entry_point="extension:ExeExtension",
            sha256="abc123",
        ),
        files={"extension.py": EXTENSION_PY_CONTENT},
    )
    report = validate_package(zip_short_hash)
    assert not report.is_valid
    assert any("sha256" in e and "长度" in e for e in report.messages)

    # 非十六进制字符的 sha256（长度为 64）
    zip_bad_hex = tmp_path / "bad_hex.ppcext.zip"
    _create_zip(
        zip_bad_hex,
        manifest=_valid_manifest(
            package_type="executable",
            entry_point="extension:ExeExtension",
            sha256="z" * 64,
        ),
        files={"extension.py": EXTENSION_PY_CONTENT},
    )
    report = validate_package(zip_bad_hex)
    assert not report.is_valid
    assert any("sha256" in e and "十六进制" in e for e in report.messages)


def test_validate_executable_requires_entry_point(tmp_path: Path):
    """executable 类型：缺少 entry_point 失败。"""
    zip_no_ep = tmp_path / "no_ep.ppcext.zip"
    _create_zip(
        zip_no_ep,
        manifest=_valid_manifest(
            package_type="executable",
            entry_point="",
            sha256=_VALID_SHA256,
        ),
        files={"extension.py": EXTENSION_PY_CONTENT},
    )
    report = validate_package(zip_no_ep)
    assert not report.is_valid
    assert any("entry_point" in e and "executable" in e for e in report.messages)


def test_validate_invalid_package_type(tmp_path: Path):
    """package_type 不在 python / executable 中时报错。"""
    zip_bad_type = tmp_path / "bad_type.ppcext.zip"
    _create_zip(
        zip_bad_type,
        manifest=_valid_manifest(package_type="foo"),
        files={"extension.py": EXTENSION_PY_CONTENT},
    )
    report = validate_package(zip_bad_type)
    assert not report.is_valid
    assert any("package_type" in e and "foo" in e for e in report.messages)


def test_validate_uppercase_sha256_accepted(tmp_path: Path):
    """executable 类型接受大写十六进制 sha256。"""
    uppercase_sha = "A" * 64
    zip_upper = tmp_path / "upper_hash.ppcext.zip"
    _create_zip(
        zip_upper,
        manifest=_valid_manifest(
            package_type="executable",
            entry_point="extension:ExeExtension",
            sha256=uppercase_sha,
        ),
        files={"extension.py": EXTENSION_PY_CONTENT},
    )
    report = validate_package(zip_upper)
    assert report.is_valid, f"大写 sha256 应通过校验，错误: {report.messages}"


# ---------------------------------------------------------------------------
# to_metadata 透传字段测试
# ---------------------------------------------------------------------------


def test_to_metadata_passes_fields():
    """to_metadata 返回的 ExtensionMetadata 包含新字段。"""
    manifest = ExtensionManifest(
        name="test-ext",
        version="1.0.0",
        description="A test extension",
        author="tester",
        extension_type="tool_integration",
        entry="extension.py",
        dependencies=["dep-a"],
        min_ppc_version="11.0.0rc1",
        tags=["tag1"],
        permissions=["filesystem_read"],
        package_type="executable",
        entry_point="extension:ExampleExtension",
        requirements=["requests", "pyyaml"],
        config_schema={"type": "object"},
        allowed_hosts=["api.example.com"],
        resource_limits={"max_cpu_percent": 50, "max_memory_mb": 256, "timeout_seconds": 30},
        sha256=_VALID_SHA256,
    )

    metadata = manifest.to_metadata()

    assert isinstance(metadata, ExtensionMetadata)
    assert metadata.name == "test-ext"
    assert metadata.extension_type == ExtensionType.TOOL_INTEGRATION
    # 新字段
    assert metadata.package_type == "executable"
    assert metadata.entry_point == "extension:ExampleExtension"
    assert metadata.requirements == ["requests", "pyyaml"]
    assert metadata.config_schema == {"type": "object"}
    assert metadata.allowed_hosts == ["api.example.com"]
    assert metadata.resource_limits == {
        "max_cpu_percent": 50,
        "max_memory_mb": 256,
        "timeout_seconds": 30,
    }
    assert metadata.sha256 == _VALID_SHA256


def test_to_metadata_defaults_for_new_fields():
    """未设置新字段时，to_metadata 返回默认值。"""
    manifest = ExtensionManifest(name="minimal-ext")
    metadata = manifest.to_metadata()

    assert isinstance(metadata, ExtensionMetadata)
    assert metadata.package_type == "python"
    assert metadata.entry_point == ""
    assert metadata.requirements == []
    assert metadata.config_schema is None
    assert metadata.allowed_hosts == []
    assert metadata.resource_limits == {}
    assert metadata.sha256 == ""
