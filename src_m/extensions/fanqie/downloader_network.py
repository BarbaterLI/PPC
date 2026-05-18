"""Tomato Novel Downloader Network - Network operations.

Contains all network-related functions: downloading, GitHub API calls, etc.
"""

import logging
import os
import platform
import shutil
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
from packaging import version

logger = logging.getLogger(__name__)


def _get_sign_key() -> bytes:
    import hashlib
    key = os.environ.get("PPC9_CONFIG_SIGN_KEY", "")
    if key:
        return key.encode("utf-8")
    machine_id = platform.node() + sys.executable
    return hashlib.sha256(machine_id.encode("utf-8")).digest()


def _compute_config_signature(content: str) -> str:
    import hmac
    import hashlib
    return hmac.new(_get_sign_key(), content.encode("utf-8"), hashlib.sha256()).hexdigest()


def get_latest_release(owner: str = "zhongbai2333", repo: str = "Tomato-Novel-Downloader") -> Dict:
    url = f"https://api.github.com/repos/{owner}/{repo}/releases/latest"
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "PPC9-Fanqie-Integration",
    }
    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"获取最新 Release 失败: {e}")


def get_releases_list(
    owner: str = "zhongbai2333",
    repo: str = "Tomato-Novel-Downloader",
    per_page: int = 10,
    page: int = 1,
) -> List[Dict]:
    url = f"https://api.github.com/repos/{owner}/{repo}/releases"
    params = {"per_page": min(per_page, 100), "page": page}
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "PPC9-Fanqie-Integration",
    }
    try:
        response = requests.get(url, headers=headers, params=params, timeout=15)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"获取 Release 列表失败: {e}")


def detect_system_info() -> Tuple[str, str]:
    system = platform.system()
    machine = platform.machine().lower()
    if hasattr(platform, "android") or "ANDROID_DATA" in os.environ:
        system = "Android"
    arch_map = {
        "x86_64": "amd64",
        "amd64": "amd64",
        "aarch64": "arm64",
        "arm64": "arm64",
        "armv7l": "arm32",
        "armv8l": "arm64",
        "i386": "386",
        "i686": "386",
    }
    architecture = arch_map.get(machine, machine)
    return system, architecture


def match_best_asset(
    assets: List[Dict],
    system: Optional[str] = None,
    arch: Optional[str] = None,
    prefer_musl: bool = False,
) -> Optional[Dict]:
    if system is None or arch is None:
        auto_system, auto_arch = detect_system_info()
        system = system or auto_system
        arch = arch or auto_arch

    candidates = []
    for asset in assets:
        name = asset["name"]
        if ("Source" in name and name.endswith(".zip")) or (
            "Source" in name and name.endswith(".tar.gz")
        ):
            continue

        if system == "Windows":
            if arch == "amd64" and "Win64" in name:
                candidates.append(("Win64", 0, asset))
            elif arch == "arm64" and "WinArm64" in name:
                candidates.append(("WinArm64", 0, asset))
        elif system == "Linux":
            if arch == "amd64":
                if "musl" in name and "amd64" in name:
                    candidates.append(("musl_amd64", 2 if prefer_musl else 1, asset))
                elif "amd64" in name and "musl" not in name:
                    candidates.append(("gnu_amd64", 0 if prefer_musl else 2, asset))
            elif arch == "arm64":
                if "musl" in name and "arm64" in name:
                    candidates.append(("musl_arm64", 2 if prefer_musl else 1, asset))
                elif "arm64" in name and "musl" not in name:
                    candidates.append(("gnu_arm64", 0 if prefer_musl else 2, asset))
        elif system in ("macOS", "Darwin"):
            if arch == "amd64" and "macOS_amd64" in name:
                candidates.append(("macOS_amd64", 0, asset))
            elif arch == "arm64" and "macOS_arm64" in name:
                candidates.append(("macOS_arm64", 0, asset))
        elif system == "Android":
            if arch == "arm64" and "Android_arm64" in name:
                candidates.append(("Android_arm64", 0, asset))
            elif arch == "arm32" and "Android_arm32" in name:
                candidates.append(("Android_arm32", 0, asset))

    if candidates:
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][2]
    return None


def convert_to_mirror_url(github_url: str, mirror: str = "gh.llkk.cc") -> str:
    return f"https://{mirror}/{github_url}"


def compare_versions(v1: str, v2: str) -> int:
    try:
        ver1 = version.parse(v1.lstrip("v"))
        ver2 = version.parse(v2.lstrip("v"))
        if ver1 < ver2:
            return -1
        elif ver1 > ver2:
            return 1
        else:
            return 0
    except Exception:
        if v1 < v2:
            return -1
        elif v1 > v2:
            return 1
        return 0


def download_file(
    url: str,
    output_path: str,
    timeout: int = 60,
    chunk_size: int = 8192,
    progress_callback=None,
) -> Tuple[bool, str]:
    try:
        response = requests.get(url, stream=True, timeout=timeout)
        response.raise_for_status()
        total_size = int(response.headers.get("content-length", 0))
        downloaded = 0
        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if progress_callback and total_size > 0:
                        progress_callback(downloaded, total_size)
        return True, url
    except requests.exceptions.RequestException as e:
        return False, str(e)


def download_with_fallback(
    url: str,
    fallback_url: str,
    output_path: str,
    timeout: int = 60,
    chunk_size: int = 8192,
    progress_callback=None,
) -> Tuple[bool, str]:
    urls_to_try = [(url, "镜像源"), (fallback_url, "官方 GitHub")]
    last_error = ""
    for try_url, source_name in urls_to_try:
        if not try_url:
            continue
        success, result = download_file(try_url, output_path, timeout, chunk_size, progress_callback)
        if success:
            return True, f"{source_name}: {try_url}"
        last_error = f"{source_name}失败: {result}"
    return False, last_error


def install_fanqie(
    use_mirror: bool = True,
    mirror: str = "gh.llkk.cc",
    prefer_musl: bool = False,
    progress_callback=None,
) -> Dict:
    from src_m.extensions.fanqie.downloader_core import (
        _get_fanqie_base_dir,
        _get_version_file_path,
        _get_exe_path,
        set_config_value
    )

    release_info = get_latest_release()
    latest_version = release_info.get("tag_name", "unknown")
    assets = release_info.get("assets", [])

    best_asset = match_best_asset(assets, prefer_musl=prefer_musl)
    if not best_asset:
        return {
            "success": False,
            "error": "未找到适合当前系统的下载资源",
            "version": latest_version,
        }

    official_url = best_asset["browser_download_url"]
    mirror_url = convert_to_mirror_url(official_url, mirror) if use_mirror else None
    filename = best_asset["name"]

    base_dir = _get_fanqie_base_dir()
    base_dir.mkdir(parents=True, exist_ok=True)

    download_path = base_dir / filename

    primary_url = mirror_url if use_mirror else official_url
    fallback = official_url if use_mirror else None

    success, msg = download_with_fallback(
        primary_url,
        fallback or "",
        str(download_path),
        timeout=120,
        progress_callback=progress_callback,
    )

    if not success:
        return {
            "success": False,
            "error": f"下载失败: {msg}",
            "version": latest_version,
        }

    extracted = _extract_and_install(download_path, base_dir)
    if not extracted:
        return {
            "success": False,
            "error": "解压或安装失败",
            "version": latest_version,
        }

    ver_file = _get_version_file_path()
    ver_file.write_text(latest_version, encoding="utf-8")

    novels_dir = _get_fanqie_base_dir() / "novels"
    novels_dir.mkdir(parents=True, exist_ok=True)
    set_config_value("save_path", str(novels_dir))

    download_path.unlink(missing_ok=True)

    return {
        "success": True,
        "version": latest_version,
        "exe_path": str(_get_exe_path()),
        "download_source": msg,
    }


def _extract_and_install(archive_path: Path, target_dir: Path) -> bool:
    from src_m.utils.files import safe_extract_zip

    system = platform.system()
    exe_path = _get_exe_path()

    if archive_path.suffix == ".zip":
        try:
            with zipfile.ZipFile(str(archive_path), "r") as zf:
                for info in zf.infolist():
                    if info.filename.endswith(".exe") or (
                        not info.filename.endswith(".exe")
                        and not info.is_dir()
                        and "/" not in info.filename
                        and "." not in Path(info.filename).suffix
                    ):
                        safe_name = Path(info.filename).name
                        dest = target_dir / safe_name
                        dest_parent = dest.resolve().parent
                        target_resolved = target_dir.resolve()
                        if not str(dest.resolve()).startswith(str(target_resolved)):
                            continue
                        dest_parent.mkdir(parents=True, exist_ok=True)
                        with zf.open(info) as src, open(dest, "wb") as dst:
                            dst.write(src.read())
                        if system != "Windows" and dest.exists():
                            dest.chmod(0o755)
                        if dest != exe_path:
                            shutil.move(str(dest), str(exe_path))
                        return True
                safe_extract_zip(zf, target_dir)
                for f in target_dir.iterdir():
                    if f.is_file() and f.suffix in (".exe", "") and f != exe_path:
                        if system != "Windows":
                            f.chmod(0o755)
                        shutil.move(str(f), str(exe_path))
                        return True
            return False
        except Exception as e:
            logger.error(f"解压失败: {e}")
            return False
    elif archive_path.suffix in (".gz", ".xz", ".bz2") or ".tar." in archive_path.name:
        import tarfile

        try:
            with tarfile.open(str(archive_path)) as tf:
                for member in tf.getmembers():
                    if member.isfile():
                        safe_name = Path(member.name).name
                        dest = target_dir / safe_name
                        dest_resolved = dest.resolve()
                        target_resolved = target_dir.resolve()
                        if not str(dest_resolved).startswith(str(target_resolved)):
                            continue
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        with tf.extractfile(member) as src, open(dest, "wb") as dst:
                            if src:
                                dst.write(src.read())
                        if system != "Windows" and dest.exists():
                            dest.chmod(0o755)
                        if dest != exe_path:
                            shutil.move(str(dest), str(exe_path))
                        return True
            return False
        except Exception as e:
            logger.error(f"解压 tar 失败: {e}")
            return False
    else:
        try:
            shutil.move(str(archive_path), str(exe_path))
            if system != "Windows":
                exe_path.chmod(0o755)
            return True
        except Exception as e:
            logger.error(f"安装失败: {e}")
            return False


def update_fanqie(
    use_mirror: bool = True,
    mirror: str = "gh.llkk.cc",
    prefer_musl: bool = False,
    progress_callback=None,
) -> Dict:
    from src_m.extensions.fanqie.downloader_core import get_installed_version

    current_version = get_installed_version()
    try:
        release_info = get_latest_release()
    except RuntimeError as e:
        return {"success": False, "error": str(e), "action": "check_failed"}

    latest_version = release_info.get("tag_name", "unknown")

    if current_version and compare_versions(current_version, latest_version) >= 0:
        return {
            "success": True,
            "action": "already_latest",
            "current_version": current_version,
            "latest_version": latest_version,
            "message": f"已是最新版本 ({current_version})",
        }

    result = install_fanqie(
        use_mirror=use_mirror,
        mirror=mirror,
        prefer_musl=prefer_musl,
        progress_callback=progress_callback,
    )
    result["action"] = "updated"
    result["previous_version"] = current_version
    return result


def check_update() -> Dict:
    from src_m.extensions.fanqie.downloader_core import get_installed_version

    current_version = get_installed_version()
    try:
        release_info = get_latest_release()
    except RuntimeError as e:
        return {"available": False, "error": str(e)}

    latest_version = release_info.get("tag_name", "unknown")
    has_update = not current_version or compare_versions(current_version, latest_version) < 0

    return {
        "available": has_update,
        "current_version": current_version,
        "latest_version": latest_version,
        "installed": is_installed(),
        "release_notes": release_info.get("body", "")[:500] if has_update else "",
    }


def uninstall_fanqie() -> bool:
    from src_m.extensions.fanqie.downloader_core import _get_fanqie_base_dir

    base_dir = _get_fanqie_base_dir()
    if not base_dir.exists():
        return True
    try:
        shutil.rmtree(str(base_dir))
        logger.info("番茄小说下载器已卸载")
        return True
    except Exception as e:
        logger.error(f"卸载失败: {e}")
        return False


def _get_fanqie_base_dir() -> Path:
    from src_m.extensions.fanqie.downloader_core import _get_fanqie_base_dir as get_base
    return get_base()


def _get_exe_path() -> Path:
    from src_m.extensions.fanqie.downloader_core import _get_exe_path as get_exe
    return get_exe()


def is_installed() -> bool:
    from src_m.extensions.fanqie.downloader_core import is_installed as check_installed
    return check_installed()


import sys
