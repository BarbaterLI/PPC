"""blb2txt 扩展 —— Extension + ToolIntegration 实现。

定位
----
**``ppc split`` 的输入格式增强器，同时是独立命令**：

* 宿主 ``ppc split`` 只接受 ``.txt``；本扩展让 EPUB / MOBI / AZW3 / FB2 /
  DOCX / PDF / CHM 等 **34 种格式**直接进入抽取与分章流程；
* ``ppc ext blb2txt extract`` —— 任意格式 → 干净 txt（可直接喂 ``ppc convert``）；
* ``ppc ext blb2txt split``   —— 任意格式 → 分章 txt 目录（等价于宿主 split 的
  产物形态，但输入不受格式限制，且可直接利用文档内嵌目录）。

产物治理
--------
blb2txt 的原始产物恒带 **UTF-8 BOM + CRLF 行尾**（实测），本扩展默认归一化
为 **UTF-8 无 BOM + LF**，避免 BOM 参与 PPC 分段时的字符计数与标题匹配。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from src.extensions.base import (
    Extension,
    ExtensionMetadata,
    ExtensionPermission,
    ExtensionType,
    ToolIntegration,
)

from .extractor import (
    SUPPORTED_FORMATS,
    Blb2TxtError,
    Blb2TxtNotFoundError,
    Blb2TxtRunner,
    _discard_new_files,
    _snapshot,
    find_blb2txt,
)

logger = logging.getLogger(__name__)

#: 扩展名（用于命令与数据目录）。
EXTENSION_NAME = "blb2txt"

#: 扩展默认配置；用户可在 ``extensions_data/blb2txt/config.yml`` 覆盖。
DEFAULT_CONFIG: dict[str, Any] = {
    # 可执行文件位置（空 = 自动探测：项目 blb2txt/ 目录 → PATH）
    "blb2txt_path": "",
    # 单次调用超时（秒）。电子书抽取可能较慢，故默认放宽到 10 分钟。
    "timeout": 600.0,
    # 产物归一化（去 BOM、CRLF→LF）
    "normalize": True,
    # 抽取产物扩展名
    "output_extension": "txt",
    # 7z.dll 路径（从 ZIP/RAR 等压缩包内抽取文档时需要；空 = 与 exe 同目录）
    "seven_zip_dll": "",
    # 分章默认值
    "default_split_mode": "auto",
    "default_name_pattern": "%Number% - %Header%",
    "default_numbering": "none",
    "min_chapter_chars": 0,
    # 文本清洗（对应 blb2txt 的 --remove-* 系列），默认关闭以免误删正文
    "clean_excess_spaces": False,
    "clean_hyphens": False,
    "clean_linebreaks": False,
    "clean_empty_lines": False,
    "strip_brackets": False,
    "remove_page_numbers": False,
    "fix_letter_spacing": False,
    # 加密 PDF 口令
    "pdf_password": "",
}

#: 防误删：这些清洗项会移除正文内容，仅由显式开关触发。
_CLEAN_FLAG_MAP: tuple[tuple[str, str], ...] = (
    ("clean_excess_spaces", "-rs"),
    ("clean_hyphens", "-rh"),
    ("clean_linebreaks", "-rl"),
    ("clean_empty_lines", "-rp"),
    ("strip_brackets", "-rsb"),
    ("remove_page_numbers", "-rpn"),
    ("fix_letter_spacing", "-ls"),
)


class Blb2TxtExtension(Extension, ToolIntegration):
    """Balabolka 文本抽取工具集成。

    用法::

        # 抽取：任意格式 -> txt（可直接 ppc convert）
        ppc ext blb2txt extract book.epub -o ./chapters

        # 分章：任意格式 -> 章节 txt 目录（split 的格式增强版）
        ppc ext blb2txt split book.mobi -o ./chapters --mode auto

        # 状态与格式
        ppc ext blb2txt status
        ppc ext blb2txt formats
    """

    def __init__(self) -> None:
        metadata = ExtensionMetadata(
            name=EXTENSION_NAME,
            version="1.0.0",
            description="Balabolka 文本抽取/分章（34 种电子书格式 → txt）",
            author="PPC_NEXT Extension",
            extension_type=ExtensionType.TOOL_INTEGRATION,
            tags=["blb2txt", "balabolka", "extract", "split", "ebook", "epub"],
            permissions=[
                # 调用 blb2txt.exe
                ExtensionPermission.SUBPROCESS,
                # 读取输入文档与 7z.dll
                ExtensionPermission.FILESYSTEM_READ,
                # 写入抽取/分章产物
                ExtensionPermission.FILESYSTEM_WRITE,
                # 清理 auto 回退链中失败尝试留下的产物
                ExtensionPermission.FILESYSTEM_DELETE,
            ],
            config=dict(DEFAULT_CONFIG),
        )
        super().__init__(metadata)
        self._user_config: dict[str, Any] = {}
        self._runner: Blb2TxtRunner | None = None

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        await super().initialize()
        try:
            self._user_config = self.load_config() or {}
        except Exception as exc:  # noqa: BLE001
            logger.warning("blb2txt: 读取扩展配置失败，使用默认值: %s", exc)
            self._user_config = {}
        logger.info("blb2txt 扩展已初始化（可用=%s）", self.is_available())

    async def cleanup(self) -> None:
        await super().cleanup()
        logger.info("blb2txt 扩展已清理")

    # ------------------------------------------------------------------
    # 配置
    # ------------------------------------------------------------------

    @property
    def config(self) -> dict[str, Any]:
        """生效配置 = 内置默认 ← manifest/包配置 ← 用户运行期配置。"""
        merged = dict(DEFAULT_CONFIG)
        merged.update(getattr(self.metadata, "config", None) or {})
        merged.update(self._user_config or {})
        return merged

    def _cfg(self, key: str, default: Any = None) -> Any:
        return self.config.get(key, DEFAULT_CONFIG.get(key, default))

    def reload_config(self) -> dict[str, Any]:
        """重新读取用户配置并重建运行器（供 CLI ``config set`` 后调用）。"""
        try:
            self._user_config = self.load_config() or {}
        except Exception as exc:  # noqa: BLE001
            logger.warning("blb2txt: 重载配置失败: %s", exc)
        self._runner = None
        return self.config

    # ------------------------------------------------------------------
    # 运行器
    # ------------------------------------------------------------------

    def resolve_exe(self) -> Path | None:
        """定位 blb2txt.exe。"""
        return find_blb2txt(self._cfg("blb2txt_path") or None)

    def get_runner(self) -> Blb2TxtRunner:
        """返回绑定当前配置的运行器（惰性缓存）。

        异常:
            Blb2TxtNotFoundError: 未找到 blb2txt.exe。
        """
        if self._runner is not None:
            return self._runner
        exe = self.resolve_exe()
        if exe is None:
            raise Blb2TxtNotFoundError(
                "未找到 blb2txt.exe。请将 blb2txt 目录放在项目根目录下"
                "（<项目>/blb2txt/blb2txt.exe），或设置环境变量 PPC_BLB2TXT，"
                "或用 'ppc ext blb2txt config set -k blb2txt_path -v <路径>' 指定。"
            )
        self._runner = Blb2TxtRunner(exe, timeout=float(self._cfg("timeout") or 600.0))
        return self._runner

    # ------------------------------------------------------------------
    # ToolIntegration 接口
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        """blb2txt.exe 是否可用（不启动进程）。"""
        return self.resolve_exe() is not None

    def get_info(self) -> dict[str, Any]:
        """工具信息（名称、可用性、版本、路径）。"""
        exe = self.resolve_exe()
        info: dict[str, Any] = {
            "name": "Balabolka 文本抽取工具",
            "installed": exe is not None,
            "exe_path": str(exe) if exe else None,
            "version": None,
            "formats": len(SUPPORTED_FORMATS),
        }
        if exe is not None:
            try:
                info["version"] = self.get_runner().version()
            except Blb2TxtError as exc:
                logger.warning("blb2txt: 读取版本失败: %s", exc)
        return info

    def get_version(self) -> str | None:
        """返回可执行文件自报的版本首行（失败返回 ``None``）。"""
        try:
            return self.get_runner().version()
        except Blb2TxtError:
            return None

    # ------------------------------------------------------------------
    # 清洗参数
    # ------------------------------------------------------------------

    def _build_extra_args(
        self,
        *,
        clean: bool | None = None,
        password: str | None = None,
        seven_zip_dll: str | None = None,
        skip_notes: bool = False,
        extract_tables: int | None = None,
        force_type: str | None = None,
        use_ifilter: bool = False,
        extra_args: list[str] | None = None,
    ) -> list[str]:
        """把结构化选项翻译为 blb2txt 原生参数列表。"""
        args: list[str] = []

        # 清洗项：clean=True 时按配置逐项启用；clean=False 时全部禁用
        enabled = self._cfg("clean_excess_spaces") if clean is None else clean
        for key, flag in _CLEAN_FLAG_MAP:
            if clean is None:
                if self._cfg(key):
                    args.append(flag)
            elif enabled:
                args.append(flag)

        pwd = password if password is not None else self._cfg("pdf_password")
        if pwd:
            args += ["-pwd", str(pwd)]

        dll = seven_zip_dll if seven_zip_dll is not None else self._cfg("seven_zip_dll")
        if dll:
            args += ["-dll", str(dll)]

        if skip_notes:
            args.append("-sn")
        if extract_tables is not None:
            args += ["-et", str(int(extract_tables))]
        if force_type:
            args += ["-x", str(force_type)]
        if use_ifilter:
            args.append("-if")

        args += list(extra_args or [])
        return args

    # ------------------------------------------------------------------
    # 抽取
    # ------------------------------------------------------------------

    def extract_text(self, input_file: Path | str, *, clean: bool | None = None) -> str:
        """抽取单个文档为文本（不落盘）。

        参数:
            input_file: 输入文档路径。
            clean: 是否应用文本清洗；``None`` 表示按扩展配置逐项决定。

        返回:
            归一化后的文本。
        """
        extra = self._build_extra_args(clean=clean)
        return self.get_runner().extract_to_text(input_file, extra_args=extra)

    def extract(
        self,
        input_file: Path | str,
        output_dir: Path | str,
        *,
        name_pattern: str | None = None,
        combine: bool = False,
        normalize: bool | None = None,
        clean: bool | None = None,
        extra_args: list[str] | None = None,
    ) -> list[Path]:
        """抽取文档到目录，产出可直接用于 ``ppc convert`` 的 txt。

        参数:
            input_file: 输入文档路径。
            output_dir: 输出目录（自动创建）。
            name_pattern: 输出名模式（``%FileName%`` 等）。
            combine: 是否合并为单个输出文件。
            normalize: 是否归一化（``None`` 取配置，默认 ``True``）。
            clean: 是否应用文本清洗。
            extra_args: 追加的原生参数。

        返回:
            产出文件列表。
        """
        norm = self._cfg("normalize") if normalize is None else normalize
        extra = self._build_extra_args(clean=clean, extra_args=extra_args)
        return self.get_runner().extract_to_dir(
            input_file,
            output_dir,
            name_pattern=name_pattern,
            extension=str(self._cfg("output_extension") or "txt"),
            combine=combine,
            normalize=bool(norm),
            extra_args=extra,
        )

    # ------------------------------------------------------------------
    # 分章（split 增强）
    # ------------------------------------------------------------------

    #: ``auto`` 模式的回退链：先试文档内嵌目录，再试关键词生成目录，最后按尺寸。
    AUTO_CHAIN: tuple[str, ...] = ("toc", "toc_gen", "size")

    #: 仅这两个模式能从文档拿到章节标题（``%Header%`` 才可解析）。
    _HEADER_MODES: frozenset[str] = frozenset({"toc", "toc_gen"})

    def _pattern_for(self, mode: str) -> str | None:
        """按分章模式选择默认文件名模式。

        ``%Header%`` 只在 toc / toc_gen 下可解析；其余模式（size / blank /
        caps / keyword）没有章节标题，传 ``-p`` 反而会留下未解析占位符
        （实测 ``-t`` 单分片 + ``-p %Number%`` 会产出字面 ``%Number%.txt``）。

        因此非标题模式**完全不传 ``-p``**，由 blb2txt 用「<输入名> <序号>」
        自行命名（实测 ``long 1.txt``），序号再由
        :func:`~.extractor.zeropad_output_numbers` 按需补零。

        用户若显式配置过 ``default_name_pattern``（与内置默认不同）则一律沿用，
        尊重用户意图。

        参数:
            mode: 分章模式。

        返回:
            输出文件名模式；``None`` 表示不传 ``-p``。
        """
        builtin = str(DEFAULT_CONFIG["default_name_pattern"])
        configured = str(self._cfg("default_name_pattern") or "")
        if configured and configured != builtin:
            return configured
        # toc 模式的 %Number% 实测自动零填充（01..12），与宿主按名排序契合。
        return "%Number% - %Header%" if mode in self._HEADER_MODES else None

    @staticmethod
    def _snapshot_dir(directory: Path) -> dict[str, tuple[int, int]]:
        """记录目录内文件指纹（委托 extractor，供回退清理做差集）。"""
        return _snapshot(directory)

    def split(
        self,
        input_file: Path | str,
        output_dir: Path | str,
        *,
        mode: str | None = None,
        keywords: list[str] | None = None,
        size_chars: int | None = None,
        name_pattern: str | None = None,
        numbering: str | None = None,
        start_number: int = 1,
        min_part_chars: int | None = None,
        min_chapter_chars: int | None = None,
        heading_prefix: str | None = None,
        normalize: bool | None = None,
        clean: bool | None = None,
        extra_args: list[str] | None = None,
    ) -> tuple[list[Path], str]:
        """把任意受支持格式的文档切分为章节文本文件。

        ``mode="auto"``（默认）时按 :data:`AUTO_CHAIN` 依次尝试，返回实际
        生效的模式 —— 这解决了「文档无内嵌目录」时 ``-c`` 静默失败的问题。

        参数:
            input_file: 输入文档路径。
            output_dir: 输出目录（自动创建）。
            mode: 分章模式，见 :meth:`Blb2TxtRunner.split_to_dir`；``auto`` 为回退链。
            keywords: 关键词模式的关键词。
            size_chars: 尺寸模式目标字符数。
            name_pattern: 输出名模式。
            numbering: 序号位置（``none`` / ``before`` / ``after``）。
            start_number: 序号起始值。
            min_part_chars: 分片最小字符数。
            min_chapter_chars: 忽略过短章节的阈值。
            heading_prefix: 标题前缀。
            normalize: 是否归一化。
            clean: 是否应用文本清洗。
            extra_args: 追加的原生参数。

        返回:
            ``(产出文件列表, 实际生效的模式)``。

        异常:
            Blb2TxtError: 所有候选模式均失败。
        """
        norm = self._cfg("normalize") if normalize is None else normalize
        pattern = name_pattern if name_pattern is not None else self._cfg("default_name_pattern")
        num = numbering if numbering is not None else self._cfg("default_numbering")
        min_chapter = (
            min_chapter_chars
            if min_chapter_chars is not None
            else int(self._cfg("min_chapter_chars") or 0)
        )
        extra = self._build_extra_args(clean=clean, extra_args=extra_args)

        requested = (mode or self._cfg("default_split_mode") or "auto").strip().lower()
        candidates = self.AUTO_CHAIN if requested == "auto" else (requested,)
        if requested == "auto" and not size_chars:
            # auto 回退链末端的 size 模式需要一个尺寸；缺省按每章约 2 万字符。
            size_chars = 20000

        errors: list[str] = []
        # 先确保目录存在，使每轮候选的回退清理有稳定的快照基准。
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        for candidate in candidates:
            # 显式传入的 pattern 覆盖所有候选；否则按候选模式选择合适默认值
            # （%Header% 在非 toc 模式下无法解析）。
            use_pattern = pattern if name_pattern is not None else self._pattern_for(candidate)
            before = self._snapshot_dir(out_path)
            try:
                produced = self.get_runner().split_to_dir(
                    input_file,
                    output_dir,
                    mode=candidate,
                    keywords=keywords,
                    size_chars=size_chars,
                    name_pattern=use_pattern,
                    numbering=str(num or "none"),
                    start_number=start_number,
                    min_part_chars=min_part_chars,
                    min_chapter_chars=min_chapter or None,
                    heading_prefix=heading_prefix,
                    normalize=bool(norm),
                    extra_args=extra,
                )
            except Blb2TxtError as exc:
                # 失败的尝试可能已落盘部分文件（如占位符未解析的产物），
                # 必须清掉再回退，避免残留污染输出目录。
                _discard_new_files(out_path, before)
                errors.append(f"{candidate}: {exc}")
                logger.info("blb2txt: 分章模式 %s 未成功，尝试回退: %s", candidate, exc)
                continue
            return produced, candidate

        raise Blb2TxtError(
            "分章失败，已尝试模式 " + " → ".join(candidates) + "：\n  " + "\n  ".join(errors)
        )

    # ------------------------------------------------------------------
    # WebUI / CLI
    # ------------------------------------------------------------------

    def get_webui_config(self) -> dict[str, Any]:
        return {
            "icon": "Book24Regular",
            "title": "文本抽取",
            "description": "从 34 种电子书格式抽取/分章为 txt",
            "mode": "none",
        }

    def register_cli(self, app) -> None:
        """注册 ``ppc ext blb2txt ...`` 子命令。"""
        from .cli import register_cli as _register_cli

        _register_cli(app, self)


__all__ = ["DEFAULT_CONFIG", "EXTENSION_NAME", "SUPPORTED_FORMATS", "Blb2TxtExtension"]
