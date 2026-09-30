"""FileProcessor.read_file_cached 缓存命中测试。"""

from __future__ import annotations

from pathlib import Path

from src.executors.file_processor import processor as proc_mod


def _fp() -> proc_mod.FileProcessor:
    cfg = {
        "split": {"encoding_fallback": ["utf-8", "gbk"], "encoding_detect_buffer": 1024},
        "performance": {"max_file_cache_size": 50},
    }
    return proc_mod.FileProcessor(cfg)


def test_read_file_cached_auto_hits_cache(tmp_path: Path, monkeypatch):
    """``encoding='auto'`` 读取后必须按**请求键**缓存，第二次调用命中（不重读磁盘）。

    Bug: auto 分支 ``get(path, "auto")`` 查键 ``path_auto``，但读取后
    ``put(path, content, _enc)`` 以**检测到的编码**（如 gbk）存 ``path_gbk`` ——
    键不匹配 → "auto" 路径**永远缓存未命中**、每次都重读磁盘，方法的核心目的
    （缓存）失效。
    修复: put 用请求的 ``encoding``（"auto"）而非检测到的 ``_enc``。
    """
    fp = _fp()
    src = tmp_path / "a.txt"
    src.write_text("你好世界", encoding="utf-8")

    assert fp.read_file_cached(src, "auto") == "你好世界"

    # 第二次调用：若命中缓存则不再调用 read_text_auto（这里让它抛错来证明）
    def boom(*_a, **_k):  # noqa: ANN002
        raise AssertionError("缓存未命中，重复读取了磁盘")

    monkeypatch.setattr(proc_mod, "read_text_auto", boom)

    result = fp.read_file_cached(src, "auto")
    assert result == "你好世界", f"第二次 auto 调用未命中缓存（返回 {result!r}，应为 '你好世界'）"
