"""ExtensionPackageManager 单元测试。"""

import logging
import zipfile
from pathlib import Path

import yaml

from src.extensions.base import (
    ExtensionMetadata,
    ExtensionPermission,
    ExtensionType,
)
from src.extensions.package import (
    ExtensionManifest,
    ExtensionPackageManager,
    read_manifest,
    validate_package,
)

# ---------------------------------------------------------------------------
# 辅助函数：创建测试用 zip 包
# ---------------------------------------------------------------------------


def _create_zip(
    zip_path: Path,
    manifest: dict | None = None,
    files: dict[str, str] | None = None,
    prefix: str = "",
) -> Path:
    """创建测试用扩展包 zip 文件。

    参数:
        zip_path: 输出 zip 文件路径
        manifest: manifest.yml 字典内容，None 表示不写入 manifest
        files: 额外文件 {filename: content}
        prefix: zip 内文件路径前缀（如 "pkgname/"）
    """
    files = files or {}
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        if manifest is not None:
            manifest_name = f"{prefix}manifest.yml" if prefix else "manifest.yml"
            zf.writestr(manifest_name, yaml.dump(manifest, allow_unicode=True))
        for name, content in files.items():
            full_name = f"{prefix}{name}" if prefix else name
            zf.writestr(full_name, content)
    return zip_path


def _valid_manifest(name: str = "test-pkg", **overrides) -> dict:
    """生成一个有效的 manifest 字典。"""
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


EXTENSION_PY_CONTENT = '''"""Test extension."""\n\nprint("hello")\n'''


# ---------------------------------------------------------------------------
# validate_package 测试
# ---------------------------------------------------------------------------


def test_validate_package_rejects_malicious_name(tmp_path: Path):
    """包含 ../ 或特殊字符的名称被拒绝。"""
    zip_path = tmp_path / "malicious.ppcext.zip"
    manifest = _valid_manifest(name="../malicious")
    _create_zip(
        zip_path,
        manifest=manifest,
        files={"extension.py": EXTENSION_PY_CONTENT},
    )
    report = validate_package(zip_path)
    assert not report.is_valid
    assert any("Invalid extension name" in e or "alphanumeric" in e for e in report.messages)


def test_validate_package_rejects_missing_name(tmp_path: Path):
    """缺少 name 字段被拒绝。"""
    zip_path = tmp_path / "no_name.ppcext.zip"
    manifest = _valid_manifest()
    del manifest["name"]
    _create_zip(
        zip_path,
        manifest=manifest,
        files={"extension.py": EXTENSION_PY_CONTENT},
    )
    report = validate_package(zip_path)
    assert not report.is_valid
    assert any("name" in e and "缺少" in e for e in report.messages)


def test_validate_package_rejects_missing_version(tmp_path: Path):
    """缺少 version 字段被拒绝。"""
    zip_path = tmp_path / "no_version.ppcext.zip"
    manifest = _valid_manifest()
    del manifest["version"]
    _create_zip(
        zip_path,
        manifest=manifest,
        files={"extension.py": EXTENSION_PY_CONTENT},
    )
    report = validate_package(zip_path)
    assert not report.is_valid
    assert any("version" in e and "缺少" in e for e in report.messages)


def test_validate_package_rejects_missing_entry(tmp_path: Path):
    """缺少 entry 字段被拒绝。"""
    zip_path = tmp_path / "no_entry.ppcext.zip"
    manifest = _valid_manifest()
    del manifest["entry"]
    _create_zip(
        zip_path,
        manifest=manifest,
        files={"extension.py": EXTENSION_PY_CONTENT},
    )
    report = validate_package(zip_path)
    assert not report.is_valid
    assert any("entry" in e and "缺少" in e for e in report.messages)


def test_validate_package_accepts_valid_name(tmp_path: Path):
    """有效名称（字母、数字、下划线、连字符）被接受。"""
    valid_names = ["my-ext", "my_ext", "MyExt123", "a-b_c", "Extension1"]
    for name in valid_names:
        zip_path = tmp_path / f"{name}.ppcext.zip"
        manifest = _valid_manifest(name=name)
        _create_zip(
            zip_path,
            manifest=manifest,
            files={"extension.py": EXTENSION_PY_CONTENT},
        )
        report = validate_package(zip_path)
        assert report.is_valid, f"名称 '{name}' 应该被接受，错误: {report.messages}"


# ---------------------------------------------------------------------------
# 路径遍历防护测试
# ---------------------------------------------------------------------------


def test_path_traversal_blocked(tmp_path: Path):
    """提取时 zip 包内含 ../ 的文件被跳过，不会写到目标目录之外。"""
    zip_path = tmp_path / "evil.ppcext.zip"
    manifest = _valid_manifest(name="evil-pkg")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("evil-pkg/manifest.yml", yaml.dump(manifest, allow_unicode=True))
        zf.writestr("evil-pkg/extension.py", EXTENSION_PY_CONTENT)
        # 路径遍历尝试
        zf.writestr("evil-pkg/../../../../../../../../evil_traversal.txt", "malicious")

    ext_dir = tmp_path / "extensions"
    pm = ExtensionPackageManager(extensions_dir=ext_dir)
    result = pm.install_package(zip_path, force=True)
    assert result["success"], f"安装应成功: {result.get('error')}"

    # 恶意文件不应存在于目标目录之外
    assert not (tmp_path / "evil_traversal.txt").exists()
    # 搜索整个 tmp_path 确认 evil_traversal.txt 没有被创建
    found = list(tmp_path.rglob("evil_traversal.txt"))
    assert found == [], f"路径遍历文件不应被解压，但找到: {found}"

    # 正常文件应被正确解压
    target = ext_dir / "evil-pkg"
    assert (target / "manifest.yml").exists()
    assert (target / "extension.py").exists()


# ---------------------------------------------------------------------------
# install / uninstall / list / info 测试
# ---------------------------------------------------------------------------


def test_install_package(tmp_path: Path):
    """安装一个有效的扩展包。"""
    zip_path = tmp_path / "test_pkg.ppcext.zip"
    manifest = _valid_manifest(name="test-pkg")
    _create_zip(
        zip_path,
        manifest=manifest,
        files={"extension.py": EXTENSION_PY_CONTENT},
    )

    ext_dir = tmp_path / "extensions"
    pm = ExtensionPackageManager(extensions_dir=ext_dir)
    result = pm.install_package(zip_path)
    assert result["success"], f"安装失败: {result.get('error')}"
    assert result["name"] == "test-pkg"
    assert result["version"] == "1.0.0"
    assert result["error"] is None

    target = ext_dir / "test-pkg"
    assert target.exists()
    assert (target / "manifest.yml").exists()
    assert (target / "extension.py").exists()
    assert (target / ".installed").exists()


def test_install_package_already_installed(tmp_path: Path):
    """重复安装未使用 force 时应失败。"""
    zip_path = tmp_path / "test_pkg.ppcext.zip"
    manifest = _valid_manifest(name="test-pkg")
    _create_zip(
        zip_path,
        manifest=manifest,
        files={"extension.py": EXTENSION_PY_CONTENT},
    )

    ext_dir = tmp_path / "extensions"
    pm = ExtensionPackageManager(extensions_dir=ext_dir)

    result1 = pm.install_package(zip_path)
    assert result1["success"]

    result2 = pm.install_package(zip_path)
    assert not result2["success"]
    assert "已安装" in result2["error"]
    assert "force" in result2["error"]


def test_uninstall_package(tmp_path: Path):
    """卸载已安装的扩展包。"""
    zip_path = tmp_path / "test_pkg.ppcext.zip"
    manifest = _valid_manifest(name="test-pkg")
    _create_zip(
        zip_path,
        manifest=manifest,
        files={"extension.py": EXTENSION_PY_CONTENT},
    )

    ext_dir = tmp_path / "extensions"
    pm = ExtensionPackageManager(extensions_dir=ext_dir)
    pm.install_package(zip_path)
    assert (ext_dir / "test-pkg").exists()

    result = pm.uninstall_package("test-pkg")
    assert result["success"], f"卸载失败: {result.get('error')}"
    assert result["name"] == "test-pkg"
    assert result["error"] is None
    assert not (ext_dir / "test-pkg").exists()


def test_uninstall_nonexistent(tmp_path: Path):
    """卸载不存在的扩展包应失败。"""
    ext_dir = tmp_path / "extensions"
    pm = ExtensionPackageManager(extensions_dir=ext_dir)
    result = pm.uninstall_package("nonexistent-pkg")
    assert not result["success"]
    assert "未安装" in result["error"]


def test_list_packages(tmp_path: Path):
    """列出所有已安装的扩展包。"""
    ext_dir = tmp_path / "extensions"
    pm = ExtensionPackageManager(extensions_dir=ext_dir)

    # 初始为空
    assert pm.list_packages() == []

    # 安装两个包
    for name in ("pkg-a", "pkg-b"):
        zip_path = tmp_path / f"{name}.ppcext.zip"
        manifest = _valid_manifest(name=name, version="2.0.0")
        _create_zip(
            zip_path,
            manifest=manifest,
            files={"extension.py": EXTENSION_PY_CONTENT},
        )
        pm.install_package(zip_path)

    packages = pm.list_packages()
    assert len(packages) == 2
    names = {p["name"] for p in packages}
    assert names == {"pkg-a", "pkg-b"}
    for p in packages:
        assert p["version"] == "2.0.0"
        assert p["enabled"] is True
        assert "path" in p


def test_get_package_info(tmp_path: Path):
    """获取已安装扩展包的详细信息。"""
    zip_path = tmp_path / "test_pkg.ppcext.zip"
    manifest = _valid_manifest(name="test-pkg", description="A test package")
    _create_zip(
        zip_path,
        manifest=manifest,
        files={"extension.py": EXTENSION_PY_CONTENT},
    )

    ext_dir = tmp_path / "extensions"
    pm = ExtensionPackageManager(extensions_dir=ext_dir)
    pm.install_package(zip_path)

    info = pm.get_package_info("test-pkg")
    assert info is not None
    assert info["name"] == "test-pkg"
    assert info["version"] == "1.0.0"
    assert info["description"] == "A test package"
    assert info["enabled"] is True
    assert "path" in info
    assert "file_list" in info
    assert "manifest.yml" in info["file_list"]
    assert "extension.py" in info["file_list"]


def test_get_package_info_nonexistent(tmp_path: Path):
    """获取不存在扩展包的信息返回 None。"""
    ext_dir = tmp_path / "extensions"
    pm = ExtensionPackageManager(extensions_dir=ext_dir)
    assert pm.get_package_info("nonexistent") is None


# ---------------------------------------------------------------------------
# create_template 测试
# ---------------------------------------------------------------------------


def test_create_template(tmp_path: Path):
    """创建新的扩展模板。"""
    pm = ExtensionPackageManager(extensions_dir=tmp_path / "extensions")
    zip_path = pm.create_template("my-template", output_dir=tmp_path)

    assert zip_path.exists()
    assert zip_path.name == "my-template.ppcext.zip"

    # 验证 zip 内容
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        assert "manifest.yml" in names
        assert "extension.py" in names

        manifest_data = yaml.safe_load(zf.read("manifest.yml"))
        assert manifest_data["name"] == "my-template"
        assert manifest_data["version"] == "1.0.0"
        assert manifest_data["entry"] == "extension.py"

        ext_content = zf.read("extension.py").decode("utf-8")
        assert "MyTemplateExtension" in ext_content
        assert "class MyTemplateExtension" in ext_content


# ---------------------------------------------------------------------------
# force 安装/卸载 测试
# ---------------------------------------------------------------------------


def test_install_with_force(tmp_path: Path):
    """force=True 时覆盖安装已存在的扩展包。"""
    ext_dir = tmp_path / "extensions"
    pm = ExtensionPackageManager(extensions_dir=ext_dir)

    # 第一次安装 v1.0.0
    zip_v1 = tmp_path / "pkg_v1.ppcext.zip"
    _create_zip(
        zip_v1,
        manifest=_valid_manifest(name="force-pkg", version="1.0.0"),
        files={"extension.py": "# v1"},
    )
    result1 = pm.install_package(zip_v1)
    assert result1["success"]
    assert result1["version"] == "1.0.0"

    # 不带 force 重新安装应失败
    result_no_force = pm.install_package(zip_v1)
    assert not result_no_force["success"]

    # 带 force 安装 v2.0.0
    zip_v2 = tmp_path / "pkg_v2.ppcext.zip"
    _create_zip(
        zip_v2,
        manifest=_valid_manifest(name="force-pkg", version="2.0.0"),
        files={"extension.py": "# v2"},
    )
    result_force = pm.install_package(zip_v2, force=True)
    assert result_force["success"], f"强制安装应成功: {result_force.get('error')}"
    assert result_force["version"] == "2.0.0"

    info = pm.get_package_info("force-pkg")
    assert info["version"] == "2.0.0"


def test_uninstall_with_force(tmp_path: Path):
    """force=True 时即使有依赖也能卸载扩展包。"""
    ext_dir = tmp_path / "extensions"
    pm = ExtensionPackageManager(extensions_dir=ext_dir)

    # 安装基础包 base-pkg
    zip_base = tmp_path / "base.ppcext.zip"
    _create_zip(
        zip_base,
        manifest=_valid_manifest(name="base-pkg"),
        files={"extension.py": EXTENSION_PY_CONTENT},
    )
    pm.install_package(zip_base)

    # 安装依赖 base-pkg 的 dependent-pkg
    zip_dep = tmp_path / "dependent.ppcext.zip"
    _create_zip(
        zip_dep,
        manifest=_valid_manifest(name="dependent-pkg", dependencies=["base-pkg"]),
        files={"extension.py": EXTENSION_PY_CONTENT},
    )
    pm.install_package(zip_dep)

    # 不带 force 卸载 base-pkg 应失败（被依赖）
    result_no_force = pm.uninstall_package("base-pkg")
    assert not result_no_force["success"]
    assert "依赖" in result_no_force["error"]
    assert (ext_dir / "base-pkg").exists()

    # 带 force 卸载 base-pkg 应成功
    result_force = pm.uninstall_package("base-pkg", force=True)
    assert result_force["success"], f"强制卸载应成功: {result_force.get('error')}"
    assert not (ext_dir / "base-pkg").exists()


# ---------------------------------------------------------------------------
# read_manifest 测试
# ---------------------------------------------------------------------------


def test_read_manifest(tmp_path: Path):
    """read_manifest 从 zip 包读取 manifest.yml。"""
    zip_path = tmp_path / "test_pkg.ppcext.zip"
    manifest = _valid_manifest(name="read-test", version="3.1.4")
    _create_zip(
        zip_path,
        manifest=manifest,
        files={"extension.py": EXTENSION_PY_CONTENT},
    )

    result = read_manifest(zip_path)
    assert isinstance(result, ExtensionManifest)
    assert result.name == "read-test"
    assert result.version == "3.1.4"


def test_read_manifest_missing(tmp_path: Path):
    """manifest.yml 缺失时抛出 ValueError。"""
    zip_path = tmp_path / "no_manifest.ppcext.zip"
    _create_zip(
        zip_path,
        manifest=None,
        files={"extension.py": EXTENSION_PY_CONTENT},
    )
    try:
        read_manifest(zip_path)
        raise AssertionError("应抛出 ValueError")
    except ValueError as e:
        assert "manifest.yml" in str(e)


def test_read_manifest_nonexistent_file(tmp_path: Path):
    """zip 文件不存在时抛出 ValueError。"""
    zip_path = tmp_path / "nonexistent.ppcext.zip"
    try:
        read_manifest(zip_path)
        raise AssertionError("应抛出 ValueError")
    except ValueError as e:
        assert "不存在" in str(e)


# ---------------------------------------------------------------------------
# ExtensionManifest.to_metadata() 转换测试
# ---------------------------------------------------------------------------


def test_to_metadata_all_fields_valid():
    """所有字段有效时正确转换为 ExtensionMetadata。"""
    manifest = ExtensionManifest(
        name="test-ext",
        version="2.1.0",
        description="A test extension",
        author="tester",
        extension_type="tool_integration",
        entry="extension.py",
        dependencies=["dep-a"],
        min_ppc_version="11.0.0rc1",
        tags=["tag1", "tag2"],
        permissions=["filesystem_read", "network"],
    )

    metadata = manifest.to_metadata()

    assert isinstance(metadata, ExtensionMetadata)
    assert metadata.name == "test-ext"
    assert metadata.version == "2.1.0"
    assert metadata.description == "A test extension"
    assert metadata.author == "tester"
    assert metadata.extension_type == ExtensionType.TOOL_INTEGRATION
    assert metadata.tags == ["tag1", "tag2"]
    assert metadata.dependencies == ["dep-a"]
    assert metadata.permissions == [
        ExtensionPermission.FILESYSTEM_READ,
        ExtensionPermission.NETWORK,
    ]


def test_to_metadata_invalid_permission_skipped(caplog):
    """无效的权限字符串被跳过并记录警告，不崩溃。"""
    manifest = ExtensionManifest(
        name="test-ext",
        permissions=["filesystem_read", "invalid_perm", "network"],
    )

    with caplog.at_level(logging.WARNING):
        metadata = manifest.to_metadata()

    assert metadata.permissions == [
        ExtensionPermission.FILESYSTEM_READ,
        ExtensionPermission.NETWORK,
    ]
    assert any("invalid_perm" in r.message for r in caplog.records)


def test_to_metadata_invalid_extension_type_skipped(caplog):
    """无效的 extension_type 被跳过并记录警告，不崩溃。"""
    manifest = ExtensionManifest(
        name="test-ext",
        extension_type="not_a_real_type",
    )

    with caplog.at_level(logging.WARNING):
        metadata = manifest.to_metadata()

    assert metadata.extension_type is None
    assert any("not_a_real_type" in r.message for r in caplog.records)


def test_to_metadata_defaults_when_fields_empty():
    """空字段使用默认值，转换不崩溃。"""
    manifest = ExtensionManifest(name="minimal-ext")

    metadata = manifest.to_metadata()

    assert isinstance(metadata, ExtensionMetadata)
    assert metadata.name == "minimal-ext"
    assert metadata.version == "1.0.0"
    assert metadata.extension_type == ExtensionType.TOOL_INTEGRATION
    assert metadata.permissions == []
    assert metadata.tags == []
    assert metadata.dependencies == []
