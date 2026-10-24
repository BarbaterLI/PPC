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
每个缓存目录 ``.cache/{stem}/`` 下维护两个文件，逐段记录
``(index, size, text_sha1, sig)``：

* ``journal.jsonl`` —— **append-only** 日志，``record()`` 每次只追加一行
  （O(1)，无需重写既有条目）；
* ``journal.json`` —— 归并后的快照，原子落盘（tmp + ``os.replace``），
  仅在「首次记录」与「下次 ``open()`` 归并日志后压实」时写入。

一次运行内 n 段 → 1 次快照 + n 次追加（O(n) I/O），而非此前的每段重写
整份 JSON（O(n²) 写入，且全部同步阻塞事件循环）。

* **崩溃一致性** —— ``record()`` 只在段合成成功返回、文件已完整落盘后调用；
  追加行因此在段文件之后落盘，崩溃可能留下的半行由 :meth:`open` 逐行解析时
  丢弃。因此 **journal 有条目 ⟹ 段文件完整**，条目缺失/大小不符 ⟹ 保守地
  重新合成。
* **内容寻址** —— ``verify()`` 比对当前分段文本的 sha1 与记录值；分段配置或
  文本一变，哈希不匹配即判为陈旧并重新合成，杜绝错位复用。
* **O(1) 命中** —— 命中判断只需一次 ``stat()`` + 哈希比对，替代整段解码。
* **向后兼容** —— 无 journal 的 11.x 缓存退回解码探针路径；仅有旧格式
  ``journal.json``（无日志）的缓存照常读取。

``journal.json`` / ``journal.jsonl`` 不匹配 ``*_seg_*.mp3`` 通配符，不影响
``rebuild_from_cache`` 等基于段文件名的扫描；成功合并后整个 ``cache_dir`` 被
``rmtree``，日志随之清理。
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
# append-only 增量日志：每段一行，避免每段重写整份快照（O(n²) → O(n)）
JOURNAL_LOG_NAME = "journal.jsonl"


def _text_sha1(text: str) -> str:
    """分段文本内容哈希（UTF-8 sha1 hex）。"""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _audio_sig(
    text: str,
    voice: str | None,
    rate: str | None,
    volume: str | None,
) -> str:
    """音频指纹：文本 + 音色 + 影响波形的合成参数（rate/volume）。

    触发条件：仅改 ``tts.voice``（或 rate/volume）而分段文本与输出字节数
    恰好不变（同长度的 TTS 输出并不罕见）。
    后果（旧实现）：命中判据只比对「文本 + 字节数」→ 复用上一音色的段文件，
    整书出现前后音色不一致。把这些参数纳入指纹后即判为失配、重新合成。

    以 ``\x00`` 分隔各字段（文本中不会出现该字符），避免拼接歧义。
    """
    h = hashlib.sha1()
    for part in (text, voice or "", rate or "", volume or ""):
        h.update(part.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


@dataclass
class JournalEntry:
    """单段记录：索引 + 落盘字节数 + 文本哈希 + 音频指纹。"""

    index: int
    size: int
    text_sha1: str
    # 音频指纹（文本 + voice + rate + volume）。空串 = 旧格式条目（无该字段），
    # 一律视为失配 → 重新合成，避免沿用无法校验音色的旧段。
    sig: str = ""


class SegmentJournal:
    """段复用日志（每缓存目录一份，条目按段索引寻址）。

    构造后调用 :meth:`open` 加载既有条目；:meth:`verify` 判命中，
    :meth:`record` 在段成功后追加一行增量日志（首次记录/日志追加失败时
    才写整份快照）。所有写操作同步无 ``await``，故在 asyncio 单线程下天然安全。
    """

    def __init__(self, path: Path, entries: dict[int, JournalEntry] | None = None) -> None:
        self._path = path
        self._log_path = path.with_name(JOURNAL_LOG_NAME)
        self._entries: dict[int, JournalEntry] = dict(entries or {})
        # 快照（整份 JSON）写入次数：供测试断言其为 O(1) 而非每段一次。
        self._snapshot_writes = 0

    # ------------------------------------------------------------------ 打开 / 持久化

    @classmethod
    def open(cls, cache_dir: Path) -> SegmentJournal:
        """打开（不存在则返回空）``cache_dir/journal.json`` 并归并增量日志。"""
        path = cache_dir / JOURNAL_NAME
        entries: dict[int, JournalEntry] = {}
        try:
            if path.exists():
                raw = json.loads(path.read_text(encoding="utf-8"))
                if raw.get("version") == JOURNAL_VERSION:
                    for item in raw.get("entries", []):
                        idx = int(item["i"])
                        entries[idx] = JournalEntry(
                            idx,
                            int(item["size"]),
                            str(item["sha1"]),
                            # 旧快照无 ``sig`` 字段 → 空串 → 恒失配（重合成）
                            str(item.get("sig") or ""),
                        )
                else:
                    logger.debug("journal 版本不匹配，忽略: %s", path)
        except Exception:  # noqa: BLE001  损坏的日志只当空处理（保守重合成）
            logger.debug("journal 加载失败，按空处理: %s", path, exc_info=True)
            entries = {}

        journal = cls(path, entries)
        if journal._merge_log():
            # 归并成功后立即压实：下次启动只读一份快照 + 空日志。
            # 若在「写完快照」与「删日志」之间崩溃，日志内容与快照一致，
            # 重放是幂等的，不会产生错误条目。
            journal._flush()
            try:
                journal._log_path.unlink()
            except OSError:
                logger.debug("journal 日志清理失败（不影响正确性）: %s", journal._log_path)
        return journal

    def _merge_log(self) -> bool:
        """把增量日志逐行重放到内存条目；返回是否合并到任何有效行。

        崩溃可能留下只写了一半的末行 → 该行解析失败即丢弃（保守地重新合成），
        前序完整行仍然生效，因此「有条目 ⟹ 段文件完整」的语义不被破坏。
        """
        if not self._log_path.exists():
            return False
        applied = False
        try:
            with self._log_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                        idx = int(item["i"])
                        self._entries[idx] = JournalEntry(
                            idx,
                            int(item["size"]),
                            str(item["sha1"]),
                            str(item.get("sig") or ""),  # 旧日志行无该字段 → 失配
                        )
                        applied = True
                    except (ValueError, KeyError, TypeError):
                        logger.debug("journal 日志存在损坏行，已跳过: %s", line[:80])
                        continue
        except OSError:
            logger.debug("journal 日志读取失败，按快照处理: %s", self._log_path, exc_info=True)
            return False
        return applied

    @property
    def path(self) -> Path:
        return self._path

    @property
    def snapshot_writes(self) -> int:
        """整份快照（``journal.json``）的写入次数（诊断/测试用）。"""
        return self._snapshot_writes

    @property
    def count(self) -> int:
        return len(self._entries)

    def completed_count(self) -> int:
        return len(self._entries)

    def has(self, index: int) -> bool:
        """该段是否已有 journal 记录（决定走内容寻址还是解码探针回退）。"""
        return index in self._entries

    # ------------------------------------------------------------------ 命中判断

    def verify(
        self,
        index: int,
        file: Path,
        text: str,
        *,
        voice: str | None = None,
        rate: str | None = None,
        volume: str | None = None,
    ) -> bool:
        """O(1) 校验段是否可复用：条目存在 + 音频指纹一致 + 文件大小一致。

        ``voice``/``rate``/``volume`` 参与指纹计算：换音色或调语速后，
        即使文本与输出字节数完全相同也不会误命中（旧判据只比对文本 + 字节
        数，会在「换音色但输出等长」时复用旧音色的段文件）。缺字段的旧条目
        指纹为空串，恒失配 → 重新合成。

        任一条件不满足即返回 ``False``（文本已变 / 合成参数已变 / 文件被
        截断或不存在 / 从未记录），调用方随后重新合成。
        """
        entry = self._entries.get(index)
        if entry is None:
            return False
        if entry.text_sha1 != _text_sha1(text):
            return False  # 内容寻址：分段文本已变 → 陈旧
        if entry.sig != _audio_sig(text, voice, rate, volume):
            return False  # 音色/语速已变（或旧条目无指纹）→ 不允许复用
        try:
            if file.stat().st_size != entry.size:
                return False  # 文件被截断/改写 → 重合成
        except OSError:
            return False
        return True

    # ------------------------------------------------------------------ 记录

    def record(
        self,
        index: int,
        file: Path,
        text: str,
        *,
        voice: str | None = None,
        rate: str | None = None,
        volume: str | None = None,
    ) -> None:
        """段合成成功后追加一行日志（失败只降级为 debug 日志）。

        触发条件：每段合成/探针命中成功后调用一次。
        改写前：每段都重写整份 ``journal.json``（n 段 → O(n²) 字节写入，
        且同步 I/O 直接阻塞事件循环）。
        改写后：只 append 一行（O(1) 字节量），首次记录补一份快照以保证
        「只认 journal.json 的旧读取方」与旧缓存兼容。
        """
        try:
            size = file.stat().st_size
        except OSError as exc:
            logger.debug("journal 记录跳过（文件不可 stat）: %s", exc)
            return
        entry = JournalEntry(
            index, size, _text_sha1(text), _audio_sig(text, voice, rate, volume)
        )
        self._entries[index] = entry
        if self._append_entry(entry):
            if not self._path.exists():
                # 新缓存目录：首次记录时同时落一份快照，保持旧格式可读。
                self._flush()
        else:
            # 日志不可写（只读目录/句柄耗尽）→ 退回整份快照，宁慢不丢记录。
            self._flush()

    def _append_entry(self, entry: JournalEntry) -> bool:
        """向 ``journal.jsonl`` 追加一条记录（单行，写入后关闭即落盘）。"""
        try:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            line = json.dumps(
                {"i": entry.index, "size": entry.size, "sha1": entry.text_sha1, "sig": entry.sig},
                ensure_ascii=False,
            )
            with self._log_path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
            return True
        except OSError as exc:
            logger.debug("journal 日志追加失败，退回整份快照: %s", exc)
            return False

    def _flush(self) -> None:
        """原子写入 ``journal.json`` 快照（tmp + ``os.replace``）。"""
        try:
            data = {
                "version": JOURNAL_VERSION,
                "updated_at": time.time(),
                "entries": [
                    {"i": e.index, "size": e.size, "sha1": e.text_sha1, "sig": e.sig}
                    for e in sorted(self._entries.values(), key=lambda e: e.index)
                ],
            }
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_name(self._path.name + ".tmp")
            tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            os.replace(str(tmp), str(self._path))
            self._snapshot_writes += 1
        except Exception as exc:  # noqa: BLE001
            logger.debug("journal 刷新失败（不影响当前运行）: %s", exc)


__all__ = ["JOURNAL_LOG_NAME", "JOURNAL_NAME", "JOURNAL_VERSION", "JournalEntry", "SegmentJournal"]
