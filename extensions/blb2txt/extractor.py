"""blb2txt.exe 子进程封装：文本抽取与分章。

实测得到的关键行为（均经真实调用验证）
--------------------------------------
* **输出目录必须预先存在**：``-v`` 指向不存在的目录时以 ``rc=1`` 失败并输出
  ``[Error] Output folder not found``，且**不会自动创建**。本模块统一先
  ``mkdir(parents=True)``。
* **``-o`` 会忽略 ``-v`` 与 ``-p``**：stdout 模式与落盘模式互斥，不可混用。
* **失败以 ``rc=1`` + ``[Error]`` 表达**；但**无参数时 ``rc=0`` 并打印 usage**，
  故不可仅凭返回码判断成功，需同时检查 stdout 是否含 ``[Error]``。
* **输出恒带 UTF-8 BOM 与 CRLF 行尾**：本模块默认归一化（去 BOM、CRLF→LF），
  使产物可直接喂给 PPC 的分段/转换管线。
* 输出编码由 ``-e`` 控制（``ansi``/``utf8``/``unicode``），默认 ``ansi``
  （简体中文 Windows 上为 GBK）。本模块固定请求 ``utf8`` 以保证可预测解码。
"""

from __future__ import annotations

import codecs
import locale
import logging
import os
import re
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

#: 环境变量覆盖 blb2txt.exe 路径。
BLB2TXT_ENV = "PPC_BLB2TXT"

#: 官方文档声明的支持格式（用于 ``formats`` 命令展示与类型提示）。
SUPPORTED_FORMATS: tuple[str, ...] = (
    "azw", "azw3", "chm", "djvu", "doc", "docx", "epub", "fb2", "fb2.zip", "fbz",
    "fb3", "html", "htm", "lit", "md", "mht", "mobi", "odp", "ods", "odt", "pdb",
    "pdf", "ppt", "pptx", "prc", "rtf", "tcr", "txt", "txtz", "wpd", "wri",
    "xls", "xlsx", "xsl", "xslx", "eml",
)

_ERROR_RE = re.compile(r"^\s*\[Error\]", re.MULTILINE)

#: 未解析的输出名占位符（``%Header%`` / ``%Number%`` 等）。
#: blb2txt 在无法解析某个变量时**不会报错**，而是把占位符原样留在文件名里
#: （实测：无标题文档用 toc 模式会产出名为 ``%Number% - %Header%.txt`` 的文件）。
#: 这是「该模式不适用于此文档」的可靠信号，必须据此触发回退。
_PLACEHOLDER_RE = re.compile(r"%[A-Za-z]+%")

_CREATE_NO_WINDOW = 0x08000000 if os.name == "nt" else 0


class Blb2TxtError(RuntimeError):
    """blb2txt 执行失败。"""


class Blb2TxtNotFoundError(Blb2TxtError):
    """未找到 blb2txt.exe。"""


def _popen_kwargs() -> dict:
    if os.name == "nt":
        return {"creationflags": _CREATE_NO_WINDOW}
    return {}


# ---------------------------------------------------------------------------
# 定位与文本归一化
# ---------------------------------------------------------------------------

def _project_root() -> Path:
    """扩展位于 ``<root>/extensions/<name>/``，故上溯两层即项目根。"""
    return Path(__file__).resolve().parents[2]


def find_blb2txt(configured: str | Path | None = None) -> Path | None:
    """按 显式配置 → 环境变量 → 项目内 ``blb2txt/`` → 扩展 bin/ → PATH 定位。

    参数:
        configured: 扩展配置中显式指定的 blb2txt.exe 路径。

    返回:
        可执行文件路径；未找到返回 ``None``。
    """
    names = ("blb2txt.exe", "blb2txt")

    if configured:
        path = Path(str(configured)).expanduser()
        if path.is_file():
            return path
        logger.warning("配置的 blb2txt 路径不存在: %s", path)

    env_path = os.environ.get(BLB2TXT_ENV)
    if env_path:
        path = Path(env_path).expanduser()
        if path.is_file():
            return path

    roots = [
        _project_root() / "blb2txt",
        _project_root() / "extensions" / "blb2txt" / "bin",
        Path(__file__).resolve().parent / "bin",
    ]
    for root in roots:
        for name in names:
            candidate = root / name
            if candidate.is_file():
                return candidate

    for name in names:
        which = shutil.which(name)
        if which:
            return Path(which)
    return None


def decode_output(raw: bytes) -> str:
    """解码 blb2txt 的 stdout（按 ``-e utf8`` 请求）。

    优先 ``utf-8-sig``（剥离 BOM）；失败时回退系统首选编码（简体中文 Windows
    为 GBK），再失败则以替换字符兜底，保证不因编码问题中断流程。

    参数:
        raw: 原始字节。

    返回:
        解码后的文本。
    """
    for enc in ("utf-8-sig", locale.getpreferredencoding(False), "utf-8", "latin-1"):
        if not enc:
            continue
        try:
            return raw.decode(enc)
        except (UnicodeDecodeError, LookupError):
            continue
    return raw.decode("utf-8", "replace")


def normalize_text(text: str) -> str:
    """归一化 blb2txt 产物：去 BOM、CRLF/CR → LF、去尾随空白行。

    参数:
        text: 原始文本。

    返回:
        归一化后的文本。
    """
    if text.startswith("\ufeff"):
        text = text[1:]
    return text.replace("\r\n", "\n").replace("\r", "\n").strip("\n") + "\n"


def normalize_file(path: Path) -> bool:
    """就地把文件归一化为 UTF-8(无 BOM) + LF 行尾。

    归一化的理由：宿主编码候选链以 ``utf-8`` 打头（见
    ``src/utils/files.py::DEFAULT_ENCODING_FALLBACK``），``utf-8`` 解码**不会**
    剥离 BOM，首章的 ``\\ufeff`` 会进入分段与合成文本。

    参数:
        path: 目标文件。

    返回:
        实际发生改写返回 ``True``；内容已规范或读写失败返回 ``False``。
    """
    try:
        raw = path.read_bytes()
    except OSError as exc:
        logger.warning("读取产物失败，跳过归一化 %s: %s", path, exc)
        return False

    target = normalize_text(decode_output(raw))

    # 变更判定必须基于**原始字节**，不能基于已被归一化的字符串，
    # 否则 BOM/CRLF 会被判定为「无需修改」而漏改。
    has_bom = raw.startswith(codecs.BOM_UTF8)
    has_cr = b"\r" in raw
    try:
        raw_text = raw.decode("utf-8-sig")  # 仅用于文本内容比对（不含换行翻译）
    except UnicodeDecodeError:
        raw_text = None

    if not has_bom and not has_cr and raw_text == target:
        return False

    try:
        path.write_text(target, encoding="utf-8", newline="\n")
        return True
    except OSError as exc:
        logger.warning("写入归一化结果失败 %s: %s", path, exc)
        return False


# ---------------------------------------------------------------------------
# 运行器
# ---------------------------------------------------------------------------

class Blb2TxtRunner:
    """封装对 ``blb2txt.exe`` 的调用（抽取、分章、能力查询）。"""

    def __init__(self, exe_path: Path | str, *, timeout: float = 600.0) -> None:
        self.exe_path = Path(exe_path)
        self.timeout = float(timeout)

    # -- 基础调用 -----------------------------------------------------------

    def _run(self, args: list[str], *, timeout: float | None = None) -> subprocess.CompletedProcess:
        cmd = [str(self.exe_path), *args]
        try:
            return subprocess.run(
                cmd,
                capture_output=True,
                timeout=timeout or self.timeout,
                **_popen_kwargs(),
            )
        except subprocess.TimeoutExpired as exc:
            raise Blb2TxtError(f"blb2txt 执行超时（>{timeout or self.timeout}s）") from exc
        except OSError as exc:
            raise Blb2TxtNotFoundError(f"无法执行 blb2txt（{self.exe_path}）: {exc}") from exc

    @staticmethod
    def _check(proc: subprocess.CompletedProcess, context: str) -> str:
        """校验返回码与 ``[Error]`` 标记，返回解码后的 stdout。"""
        stderr_text = decode_output(proc.stderr or b"").strip()
        stdout_text = decode_output(proc.stdout or b"")

        if proc.returncode != 0:
            detail = stderr_text or stdout_text.strip() or "（无输出）"
            raise Blb2TxtError(f"{context}失败（返回码 {proc.returncode}）: {detail}")

        # 无参数调用会以 rc=0 打印 usage；含 [Error] 同样是失败。
        if _ERROR_RE.search(stdout_text) or "Usage: blb2txt" in stdout_text:
            detail = (stdout_text.strip() or stderr_text or "（无输出）")[:300]
            raise Blb2TxtError(f"{context}失败: {detail}")

        return stdout_text

    # -- 能力面 -------------------------------------------------------------

    def version(self) -> str:
        """返回首行版本信息（``-h`` 输出第一行）。

        注意: 官方 readme 声明的版本号可能高于可执行文件实际版本
        （实测 readme 1.120 而 exe 报 1.119），此处以**可执行文件自报**为准。
        """
        proc = self._run(["-h"])
        text = decode_output(proc.stdout or b"") or decode_output(proc.stderr or b"")
        first = text.splitlines()[0].strip() if text else ""
        return first or "blb2txt（版本未知）"

    # -- 抽取 ---------------------------------------------------------------

    def extract_to_text(self, input_file: Path | str, *, extra_args: list[str] | None = None) -> str:
        """抽取单个文件为文本（走 stdout，不落盘）。

        参数:
            input_file: 输入文档路径。
            extra_args: 追加的 blb2txt 原生选项（如 ``["-rm"]``）。

        返回:
            归一化后的文本（UTF-8 无 BOM、LF 行尾）。

        异常:
            Blb2TxtError: 输入不存在或抽取失败。
        """
        source = Path(input_file)
        if not source.exists():
            raise Blb2TxtError(f"输入文件不存在: {source}")

        # -o 与 -v/-p 互斥：此路径不传输出目录。
        args = ["-f", str(source), "-o", "-e", "utf8"]
        args += list(extra_args or [])
        proc = self._run(args)
        return normalize_text(self._check(proc, f"抽取 {source.name}"))

    def extract_to_dir(
        self,
        input_file: Path | str,
        output_dir: Path | str,
        *,
        name_pattern: str | None = None,
        extension: str = "txt",
        combine: bool = False,
        normalize: bool = True,
        extra_args: list[str] | None = None,
    ) -> list[Path]:
        """抽取文档到输出目录（落盘为 txt）。

        参数:
            input_file: 输入文档路径。
            output_dir: 输出目录（不存在时自动创建）。
            name_pattern: 输出文件名模式，支持 ``%FileName%`` / ``%FirstLine%`` /
                ``%Header%`` / ``%Number%`` / ``%Title%``。
            extension: 输出扩展名，默认 ``txt``。
            combine: 是否合并为单个输出文件（``-u``）。
            normalize: 是否归一化产物（去 BOM、CRLF→LF）。
            extra_args: 追加的 blb2txt 原生选项。

        返回:
            产出文件列表（按名称排序）。

        异常:
            Blb2TxtError: 输入不存在、目录不可用或抽取失败。
        """
        source = Path(input_file)
        if not source.exists():
            raise Blb2TxtError(f"输入文件不存在: {source}")

        out_dir = Path(output_dir)
        # blb2txt 不会自动创建输出目录（实测报 [Error] Output folder not found）
        out_dir.mkdir(parents=True, exist_ok=True)

        args = ["-f", str(source), "-v", str(out_dir), "-e", "utf8", "-ext", extension]
        if name_pattern:
            args += ["-p", name_pattern]
        if combine:
            args += ["-u"]
        args += list(extra_args or [])

        before = _snapshot(out_dir)
        try:
            proc = self._run(args)
            self._check(proc, f"抽取 {source.name}")
        except Blb2TxtError:
            # blb2txt 在报错前可能已落盘半成品（实测会留下仅含 BOM 的空文件），
            # 必须清掉再上抛，避免污染输出目录。
            _discard_new_files(out_dir, before)
            raise
        produced = _diff_snapshot(out_dir, before)

        if not produced:
            # 少数情况下 blb2txt 静默无产物（如内容为空文档）
            raise Blb2TxtError(f"抽取 {source.name} 未产出任何文件（文档可能为空或格式不受支持）")

        _reject_unresolved_placeholders(produced, "extract")
        if normalize:
            for path in produced:
                normalize_file(path)
        return produced

    # -- 分章 ---------------------------------------------------------------

    def split_to_dir(
        self,
        input_file: Path | str,
        output_dir: Path | str,
        *,
        mode: str = "toc",
        keywords: list[str] | None = None,
        size_chars: int | None = None,
        name_pattern: str | None = None,
        numbering: str = "none",
        start_number: int = 1,
        min_part_chars: int | None = None,
        min_chapter_chars: int | None = None,
        heading_prefix: str | None = None,
        normalize: bool = True,
        extra_args: list[str] | None = None,
    ) -> list[Path]:
        """从任意受支持格式直接切分为多个文本文件（分章）。

        这是对宿主 ``ppc split`` 的增强：宿主只接受 ``.txt``，本方法可处理
        EPUB / MOBI / AZW3 / FB2 / DOCX / PDF 等 34 种格式，且分章依据可直接
        来自文档内嵌目录（``-c``），无需先把整书转成单一 txt 再二次切分。

        参数:
            input_file: 输入文档路径。
            output_dir: 输出目录（不存在时自动创建）。
            mode: 分章依据 —— ``toc``（内嵌目录）/ ``toc_gen``（关键词生成目录）/
                ``keyword``（按关键词切分并保留）/ ``keyword_remove``（切分并移除）/
                ``size``（按字符数）/ ``blank``（连续两个空行）/ ``caps``（全大写行）。
            keywords: ``keyword`` / ``keyword_remove`` 模式的关键词（区分大小写，可多个）。
            size_chars: ``size`` 模式的目标字符数（``-t``）。
            name_pattern: 输出名模式，默认 ``%Number% - %Header%``（分章场景可读性最佳）。
            numbering: 序号位置 —— ``none`` / ``before``（``-b``）/ ``after``（``-a``）。
            start_number: 序号起始值（``-n``）。
            min_part_chars: 分片最小字符数（``-m``）。
            min_chapter_chars: 忽略过短章节的阈值（``-j``，配合 toc 模式）。
            heading_prefix: 标题前缀文本（``-hh``）。
            normalize: 是否归一化产物。
            extra_args: 追加的 blb2txt 原生选项。

        返回:
            产出文件列表（按名称排序）。

        异常:
            Blb2TxtError: 参数非法、输入不存在或分割失败。
        """
        source = Path(input_file)
        if not source.exists():
            raise Blb2TxtError(f"输入文件不存在: {source}")

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        args = ["-f", str(source), "-v", str(out_dir), "-e", "utf8", "-ext", "txt"]

        mode_args = {
            "toc": ["-c"],
            "toc_gen": ["-toc"],
            "keyword": ["-k"],
            "keyword_remove": ["-r"],
            "size": ["-t"],
            "blank": ["-w"],
            "caps": ["-l"],
        }
        mode = (mode or "toc").strip().lower()
        if mode not in mode_args:
            raise Blb2TxtError(
                f"未知分章模式: {mode}（可选: {', '.join(mode_args)}）"
            )

        if mode in {"keyword", "keyword_remove"}:
            if not keywords:
                raise Blb2TxtError(f"模式 {mode} 需要至少一个 --keyword")
            flag = mode_args[mode][0]
            for kw in keywords:
                args += [flag, kw]
        elif mode == "size":
            if not size_chars or int(size_chars) <= 0:
                raise Blb2TxtError("模式 size 需要大于 0 的 --size")
            args += ["-t", str(int(size_chars))]
        else:
            args += mode_args[mode]

        if numbering == "before":
            args += ["-b"]
        elif numbering == "after":
            args += ["-a"]
        if start_number != 1:
            args += ["-n", str(int(start_number))]
        if min_part_chars:
            args += ["-m", str(int(min_part_chars))]
        if min_chapter_chars and mode in {"toc", "toc_gen"}:
            args += ["-j", str(int(min_chapter_chars))]
        if heading_prefix:
            args += ["-hh", heading_prefix]

        pattern = name_pattern
        if pattern is None and mode in {"toc", "toc_gen"}:
            pattern = "%Number% - %Header%"
        if pattern:
            args += ["-p", pattern]

        args += list(extra_args or [])

        before = _snapshot(out_dir)
        try:
            proc = self._run(args)
            self._check(proc, f"分割 {source.name}")
        except Blb2TxtError:
            # 与 extract_to_dir 一致：失败尝试留下的新建文件必须清掉，
            # 否则 auto 回退链与后续模式会在脏目录上继续工作。
            _discard_new_files(out_dir, before)
            raise
        produced = _diff_snapshot(out_dir, before)

        if not produced:
            raise Blb2TxtError(
                f"分割 {source.name} 未产出任何文件。若使用 toc 模式，"
                f"该文档可能不含内嵌目录，可改用 --mode toc_gen 或 --mode size。"
            )

        _reject_unresolved_placeholders(produced, mode)
        if normalize:
            for path in produced:
                normalize_file(path)
        return produced


def _discard_new_files(directory: Path, before: dict[str, tuple[int, int]]) -> None:
    """删除本次运行**新建**的文件；已存在的文件一律不动。

    只按「名字不在 ``before`` 中」判定。被本次运行覆盖过的既有文件**不删**：
    其原始内容已不可恢复，删除只会造成更大损失；这也避免误删用户自有文件。

    参数:
        directory: 输出目录。
        before: 运行前的 :func:`_snapshot` 结果。
    """
    after = _snapshot(directory)
    for name in after:
        if name in before:
            continue
        try:
            (directory / name).unlink()
        except OSError as exc:
            logger.debug("blb2txt: 清理失败尝试产物失败 %s: %s", name, exc)


# ---------------------------------------------------------------------------
# 目录快照（用于识别本次产出，避免误判历史文件与用户自有文件）
# ---------------------------------------------------------------------------

def _snapshot(directory: Path) -> dict[str, tuple[int, int]]:
    """记录目录内每个文件的 ``(大小, mtime_ns)``。

    仅记名字不足以识别「本次改写」：重复运行同一命令时 blb2txt 会**覆盖**
    同名产物，名字集合不变。故需记录内容指纹。
    """
    snapshot: dict[str, tuple[int, int]] = {}
    try:
        for path in directory.iterdir():
            if not path.is_file():
                continue
            try:
                stat = path.stat()
            except OSError:
                continue
            snapshot[path.name] = (stat.st_size, stat.st_mtime_ns)
    except OSError:
        return {}
    return snapshot


def _diff_snapshot(directory: Path, before: dict[str, tuple[int, int]]) -> list[Path]:
    """返回本次运行**新建或被改写**的文件。

    不再退化为「返回整个目录」——那会把用户放在同一目录里的自有文件卷进来
    （历史缺陷：重复运行会把 `notes 2.txt` 之类无关文件改名/纳入产出清单）。
    大小或 mtime 任一变即视为本次写过。

    参数:
        directory: 输出目录。
        before: 运行前的 :func:`_snapshot` 结果。

    返回:
        按名称排序的产物路径列表（可能为空）。
    """
    after = _snapshot(directory)
    touched = [
        name
        for name, fingerprint in after.items()
        if before.get(name) != fingerprint
    ]
    return [directory / name for name in sorted(touched)]


def _reject_unresolved_placeholders(produced: list[Path], mode_label: str) -> None:
    """产出文件名仍含 ``%Var%`` 占位符时判定该模式不适用。

    blb2txt 对这些变量采用「解析失败则原样保留」策略且返回码为 0，若不加
    判别会被误当作成功（进而不触发 :data:`Blb2TxtExtension.AUTO_CHAIN` 回退），
    并留下 `%Number% - %Header%.txt` 这类不可用文件名。

    实测：无标题文档用 ``-c`` 会产出名为 ``%Number% - %Header%.txt`` 的文件；
    带 ``-t`` 但内容不足一个分片时 ``-p %Number%`` 也会留下字面 ``%Number%``。

    参数:
        produced: 本次产出的文件列表。
        mode_label: 用于错误信息的分章模式名。

    异常:
        Blb2TxtError: 存在未解析占位符。
    """
    bad = [p.name for p in produced if _PLACEHOLDER_RE.search(p.name)]
    if bad:
        raise Blb2TxtError(
            f"模式 {mode_label} 的输出文件名未能解析占位符（文档缺少对应信息）: "
            f"{', '.join(bad[:3])}。该模式不适用于此文档。"
        )




__all__ = [
    "BLB2TXT_ENV",
    "SUPPORTED_FORMATS",
    "Blb2TxtError",
    "Blb2TxtNotFoundError",
    "Blb2TxtRunner",
    "decode_output",
    "find_blb2txt",
    "normalize_file",
    "normalize_text",
]
