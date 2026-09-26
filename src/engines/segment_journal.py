"""段级 Journal —— 崩溃安全的段复用与完整性记录（``ppc.tts.v1`` 基础设施）。

背景
----
11.x 的段级复用只有「文件存在 + 大小>0 + 能被解码」三重检查：

* **正确性漏洞** —— 只看文件名 ``{stem}_seg_{i:03d}.mp3``。若两次运行之间
  分段配置（``max_segment_length`` 等）或输入文本发生变化，同名段文件承载的
  **内容已不同**，却会被当作命中直接复用，产出错位拼接的音频。
* **性能开销** —— 命中判断用 ``_load_audio`` 整段解码探针，一本书上百段
  各解码一次，纯属浪费。
* **不可见** —— 崩溃后无法快速得知「已完成多少段」。

Journal 方案
------------
每个缓存目录 ``.cache/{stem}/`` 下维护 ``journal.json``，逐段记录
``(index, size, text_sha1)``，原子落盘（tmp + ``os.replace``）。

* **崩溃一致性** —— ``record()`` 只在段合成成功返回、文件已完整落盘后调用；
  ``journal.json`` 的写入本身原子。因此 **journal 有条目 ⟹ 段文件完整**，
  条目缺失/大小不符 ⟹ 保守地重新合成。
* **内容寻址** —— ``verify()`` 比对当前分段文本的 sha1 与记录值；分段配置或
  文本一变，哈希不匹配即判为陈旧并重新合成，杜绝错位复用。
* **O(1) 命中** —— 命中判断只需一次 ``stat()`` + 哈希比对，替代整段解码。
* **向后兼容** —— 无 ``journal.json`` 的 11.x 缓存退回解码探针路径。

``journal.json`` 不匹配 ``*_seg_*.mp3`` 通配符，不影响 ``rebuild_from_cache``
等基于段文件名的扫描；成功合并后整个 ``cache_dir`` 被 ``rmtree``，日志随之清理。
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

JOURNAL_VERSION = 1
JOURNAL_NAME = "journal.json"


def _text_sha1(text: str) -> str:
    """分段文本内容哈希（UTF-8 sha1 hex）。"""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


@dataclass
class JournalEntry:
    """单段记录：索引 + 落盘字节数 + 文本哈希。"""

    index: int
    size: int
    text_sha1: str


class SegmentJournal:
    """段复用日志（每缓存目录一份，条目按段索引寻址）。

    构造后调用 :meth:`open` 加载既有条目；:meth:`verify` 判命中，
    :meth:`record` 在段成功后追加并原子刷新。所有写操作同步无 ``await``，
    故在 asyncio 单线程下天然安全。
    """

    def __init__(self, path: Path, entries: dict[int, JournalEntry] | None = None) -> None:
        self._path = path
        self._entries: dict[int, JournalEntry] = dict(entries or {})

    # ------------------------------------------------------------------ 打开 / 持久化

    @classmethod
    def open(cls, cache_dir: Path) -> SegmentJournal:
        """打开（不存在则返回空）``cache_dir/journal.json``。"""
        path = cache_dir / JOURNAL_NAME
        entries: dict[int, JournalEntry] = {}
        try:
            if path.exists():
                raw = json.loads(path.read_text(encoding="utf-8"))
                if raw.get("version") == JOURNAL_VERSION:
                    for item in raw.get("entries", []):
                        idx = int(item["i"])
                        entries[idx] = JournalEntry(idx, int(item["size"]), str(item["sha1"]))
                else:
                    logger.debug("journal 版本不匹配，忽略: %s", path)
        except Exception:  # noqa: BLE001  损坏的日志只当空处理（保守重合成）
            logger.debug("journal 加载失败，按空处理: %s", path, exc_info=True)
            entries = {}
        return cls(path, entries)

    @property
    def path(self) -> Path:
        return self._path

    @property
    def count(self) -> int:
        return len(self._entries)

    def completed_count(self) -> int:
        return len(self._entries)

    def has(self, index: int) -> bool:
        """该段是否已有 journal 记录（决定走内容寻址还是解码探针回退）。"""
        return index in self._entries

    # ------------------------------------------------------------------ 命中判断

    def verify(self, index: int, file: Path, text: str) -> bool:
        """O(1) 校验段是否可复用：条目存在 + 文本哈希一致 + 文件大小一致。

        任一条件不满足即返回 ``False``（文本已变 / 文件被截断或不存在 /
        从未记录），调用方随后重新合成。
        """
        entry = self._entries.get(index)
        if entry is None:
            return False
        if entry.text_sha1 != _text_sha1(text):
            return False  # 内容寻址：分段文本已变 → 陈旧
        try:
            if file.stat().st_size != entry.size:
                return False  # 文件被截断/改写 → 重合成
        except OSError:
            return False
        return True

    # ------------------------------------------------------------------ 记录

    def record(self, index: int, file: Path, text: str) -> None:
        """段合成成功后记录并原子刷新（失败只降级为 debug 日志）。"""
        try:
            size = file.stat().st_size
        except OSError as exc:
            logger.debug("journal 记录跳过（文件不可 stat）: %s", exc)
            return
        self._entries[index] = JournalEntry(index, size, _text_sha1(text))
        self._flush()

    def _flush(self) -> None:
        """原子写入 ``journal.json``（tmp + ``os.replace``）。"""
        try:
            data = {
                "version": JOURNAL_VERSION,
                "updated_at": time.time(),
                "entries": [
                    {"i": e.index, "size": e.size, "sha1": e.text_sha1}
                    for e in sorted(self._entries.values(), key=lambda e: e.index)
                ],
            }
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_name(self._path.name + ".tmp")
            tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            os.replace(str(tmp), str(self._path))
        except Exception as exc:  # noqa: BLE001
            logger.debug("journal 刷新失败（不影响当前运行）: %s", exc)


__all__ = ["JOURNAL_NAME", "JOURNAL_VERSION", "JournalEntry", "SegmentJournal"]
