"""转换命令实现 - 批量 TTS 转换与进度追踪"""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from src.cli.output import get_output

from ...config import ConfigManager, get_preset
from ...executors import TTSExecutor
from ...reliability import create_tts_retry_policy
from ...core.exceptions import CLIError
from ...core.exceptions import ErrorCodes as E
from ..simple_progress import SimpleProgressHandler
from ._async_runner import run_cli_async

if TYPE_CHECKING:
    from src.config import PPCConfig
    from src.cli.output import OutputFormatter


@dataclass
class ConvertOptions:
    """convert 命令参数封装。"""

    input: Path
    output: Path | None
    voice: str | None
    concurrency: int | None
    preset: str
    resume: bool = False
    checkpoint: Path | None = None
    timeout_multiplier: float | None = None
    rate: str | None = None
    pitch: str | None = None
    annotate: bool = False
    annotate_only: bool = False
    annotate_model: str | None = None
    annotate_base_url: str | None = None
    annotate_api_key: str | None = None
    strip_attribution: bool = False
    keep_segment_cache: bool = False
    recursive: bool = False
    ramp_up: float | None = None
    strict: bool = False
    one: bool = False
    timeout_mode: str | None = None
    timeout: int | None = None
    tts_backend: str | None = None
    tts_proxy: str | None = None
    book: bool = False
    book_title: str | None = None
    cast: str | None = None
    # 多角色合并优化（合并同音色片段 + 字幕时间截断）
    merge_cut: bool = False
    merge_max_chars: int | None = None
    merge_max_bytes: int | None = None
    merge_max_pieces: int | None = None
    subtitles: bool = False
    # LLM 自动分配发声人
    llm_assign: bool = False
    llm_model: str | None = None
    llm_base_url: str | None = None
    llm_api_key: str | None = None
    main_roles: str | None = None
    candidate_voices: str | None = None
    save_cast: str | None = None


def _normalize_rate(rate: str) -> str:
    import re as _re

    rate_val = rate.strip()
    if _re.match(r"^\d+%$", rate_val):
        rate_val = f"+{rate_val}"
    elif not _re.match(r"^[+-]\d+%$", rate_val):
        raise CLIError(
            E.E_BUSINESS,
            f"无效的 rate 参数: '{rate}',应为 '+0%'、'+40%'、'-10%' 等格式",
            hint="rate 必须带正负号前缀和百分号,例如 --rate +40% 或 --rate -10%",
        )
    return rate_val


def _parse_role_map(value: str) -> dict[str, str]:
    """解析「角色→音色」映射：JSON 字符串 / JSON·YAML 文件 / ``角色=音色,角色2=音色2``。"""
    import json

    raw = (value or "").strip()
    if not raw:
        return {}

    if raw.startswith("{"):
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            raise CLIError(E.E_BUSINESS, f"--main-roles JSON 解析失败: {e}") from None
        if not isinstance(data, dict):
            raise CLIError(E.E_BUSINESS, "--main-roles 必须是 JSON 对象（角色 → 音色）")
        return {str(k).strip(): str(v).strip() for k, v in data.items() if k and v}

    path = Path(raw)
    if path.exists():
        text = path.read_text(encoding="utf-8")
        try:
            if path.suffix.lower() in (".yml", ".yaml"):
                import yaml

                data = yaml.safe_load(text) or {}
            else:
                data = json.loads(text)
        except Exception as e:  # noqa: BLE001
            raise CLIError(E.E_BUSINESS, f"--main-roles 文件解析失败: {e}") from None
        if isinstance(data, dict) and isinstance(data.get("roles"), dict):
            data = data["roles"]
        if not isinstance(data, dict):
            raise CLIError(E.E_BUSINESS, "--main-roles 文件应为映射（角色 → 音色）")
        return {str(k).strip(): str(v).strip() for k, v in data.items() if k and v}

    # 内联简写：角色=音色,角色2=音色2
    out: dict[str, str] = {}
    for pair in raw.split(","):
        if "=" not in pair:
            continue
        k, v = pair.split("=", 1)
        if k.strip() and v.strip():
            out[k.strip()] = v.strip()
    return out


def _normalize_pitch(pitch: str) -> str:
    """校验并归一化 ``--pitch``（与 ``--rate`` 同一套 CLIError 语义）。

    此前直接 ``config.tts.pitch = options.pitch``：``TTSConfig`` 没开
    ``validate_assignment``，非法值会一路带到合成层才炸（cast 路径报成
    「轮次级配音合成失败」，非 cast 路径直接把 ``abc`` 发给服务端）。
    """
    from ...text.cast_prosody import PITCH_LIMIT, normalize_pitch

    try:
        value = normalize_pitch(pitch)
    except ValueError as exc:
        raise CLIError(
            E.E_BUSINESS,
            f"无效的 pitch 参数: '{pitch}'，应为 '+0Hz'、'+20Hz'、'-30Hz' 等格式",
            hint="pitch 必须带正负号前缀和 Hz，例如 --pitch +20Hz 或 --pitch -30Hz",
        ) from exc

    hz = int(value[:-2])
    if abs(hz) > PITCH_LIMIT:
        raise CLIError(
            E.E_BUSINESS,
            f"pitch 超出范围: {value}（允许 -{PITCH_LIMIT}Hz ~ +{PITCH_LIMIT}Hz）",
            hint="超出范围会被服务端拒绝或产生失真，请减小偏移",
        )
    return value


def _read_text(path: Path) -> str:
    """按**与执行器相同**的策略读取章节：编码探测 + 去 BOM。

    触发条件：章节是 GBK/GB2312/UTF-16，或带 BOM。
    后果（旧实现用 utf-8 强读）：标注缓存的正文指纹与引擎侧 `read_text_auto`
    算出的不一致 → 缓存永不命中，每次运行都白跑一遍 LLM，而且送进模型的正文
    还是乱码。实测这条会让「标注」在多编码语料上完全失效。
    """
    try:
        from ...utils.files import read_text_auto

        text, _encoding = read_text_auto(path)
        return text
    except Exception:  # noqa: BLE001 - 探测失败时退回旧的宽松读取，保证不阻断
        return path.read_text(encoding="utf-8", errors="replace")


def _apply_llm_assign(
    options: "ConvertOptions",
    output_dir: Path,
    config: "PPCConfig",
    output_formatter: "OutputFormatter",
) -> None:
    """转换前用 LLM 生成 cast 并注入 ``tts.cast_file``。

    失败（未装 openai / 无 key / 网络或额度问题）只警告并降级为现有 cast，
    绝不阻断转换。
    """
    if not getattr(config.tts.llm_assign, "enabled", False):
        return

    from ...text.cast import Cast
    from ...text.llm_assign import LLMSpeakerAssignError, LLMSpeakerAssigner

    texts: list[str] = []
    if options.one:
        try:
            texts.append(_read_text(options.input))
        except OSError as e:
            output_formatter.warning(f"LLM 分配跳过：读取输入失败 {e}")
            return
    else:
        pattern = "**/*.txt" if options.recursive else "*.txt"
        for txt in sorted(options.input.glob(pattern)):
            try:
                texts.append(_read_text(txt))
            except OSError:
                continue

    if not texts:
        output_formatter.warning("LLM 分配跳过：没有可读取的文本")
        return

    base_cast = None
    if config.tts.cast_file:
        try:
            base_cast = Cast.load(config.tts.cast_file)
        except Exception:  # noqa: BLE001
            base_cast = None

    assigner = LLMSpeakerAssigner(config.tts.llm_assign)
    try:
        output_formatter.info("LLM 分配发声人：分析文本中…")
        cast = assigner.assign("\n".join(texts), base_cast=base_cast)
    except LLMSpeakerAssignError as e:
        output_formatter.warning(f"LLM 分配失败，沿用现有音色配置: {e}")
        return
    except Exception as e:  # noqa: BLE001
        output_formatter.warning(f"LLM 分配异常，沿用现有音色配置: {e}")
        return

    persist = str(getattr(config.tts.llm_assign, "persist_cast", "") or "")
    target = persist or str(output_dir / ".llm_cast.yml")
    try:
        assigner.persist(cast, target)
    except Exception as e:  # noqa: BLE001
        # 输出目录不可写时退化到系统临时目录，别让已拿到的分配结果白白丢掉
        import tempfile

        fallback = str(Path(tempfile.gettempdir()) / "ppc_llm_cast.yml")
        try:
            assigner.persist(cast, fallback)
        except Exception as e2:  # noqa: BLE001
            output_formatter.warning(f"cast 写入失败，LLM 分配不生效: {e} / {e2}")
            return
        output_formatter.warning(f"cast 写入 {target} 失败（{e}），已改写到 {fallback}")
        target = fallback
    config.tts.cast_file = target

    output_formatter.info(f"cast 已写入: {target}")
    output_formatter.info(
        f"LLM 分配完成: {len(cast.roles)} 个角色, narrator={cast.narrator or '-'}"
    )


def _split_oversized(text: str, limit: int) -> list[str]:
    """把超长章节按**段落边界**切成多个 ``<= limit`` 的块。

    只在段落边界切：引号不会跨段成对，所以每块内的 ``quote_spans`` 与整章
    一致，逐块标注后按顺序拼接即可还原整章。
    单段本身就超限时不再细切（宁可让这一次请求偏大，也不要把一句台词劈成
    两半导致引号配对错位）。
    """
    if len(text) <= limit:
        return [text]

    # 保留行尾换行，拼回去与原文一致
    paragraphs = re.split(r"(?<=\n)", text)
    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        if current and len(current) + len(para) > limit:
            chunks.append(current)
            current = para
        else:
            current += para
    if current:
        chunks.append(current)
    return chunks or [text]


def _apply_annotate(
    options: "ConvertOptions",
    output_dir: Path,
    config: "PPCConfig",
    output_formatter: "OutputFormatter",
) -> None:
    """转换前用 LLM 逐引号标注说话人，写入章节级缓存（引擎据此重建轮次）。

    **按批合并请求**：本项目按请求次数计费，逐章一次调用会烧掉大量额度。
    因此把多章拼成一次请求（单批上限 ``annotate.max_input_chars``，默认
    30 万字），标注完再按「每章引号数」拆回各章。合并文本以换行分隔，不会
    让引号跨章（成对匹配；跨章说明原文引号未闭合，该批会被剔除回退）。

    失败只警告：引擎查不到缓存就回退正则匹配器，转换不受影响。
    """
    if not getattr(config.tts.annotate, "enabled", False):
        return

    from ...text.ai_annotate import (
        AIAnnotator,
        AnnotationCache,
        AnnotationError,
        roles_signature,
    )
    from ...text.annotate import quote_spans
    from ...text.cast import Cast

    cast_path = str(getattr(config.tts, "cast_file", "") or "")
    if not cast_path:
        output_formatter.warning("AI 标注跳过：未指定 cast 文件（无法确定候选角色）")
        return
    try:
        cast = Cast.load(cast_path)
    except Exception as e:  # noqa: BLE001
        output_formatter.warning(f"AI 标注跳过：cast 加载失败 {e}")
        return

    ann = config.tts.annotate
    directory = str(getattr(ann, "cache_dir", "") or "") or str(
        Path.home() / ".cache" / "ppc" / "annotate"
    )
    cache = AnnotationCache(directory)
    annotator = AIAnnotator(ann)
    roles = cast.role_table()
    sig = roles_signature(roles)

    if options.one:
        files = [options.input]
    else:
        pattern = "**/*.txt" if options.recursive else "*.txt"
        files = sorted(options.input.glob(pattern))

    pending: list[tuple[Path, str, int]] = []   # 待标注：路径 / 正文 / 引号数
    hit = 0
    for path in files:
        try:
            text = _read_text(path)
        except OSError:
            continue
        if cache.get(text, sig) is not None:
            hit += 1
            continue
        n = len(quote_spans(text))
        if n == 0:
            hit += 1
            continue
        pending.append((path, text, n))

    if not pending:
        output_formatter.info(f"AI 标注：全部命中缓存（{hit} 章），跳过 LLM 调用")
        return

    # 按单批输入上限切批（默认 30 万字 ≈ 1M token 上下文的安全区）
    max_batch_chars = max(1000, int(getattr(ann, "max_input_chars", 300_000)))
    max_out_chars = max(1000, int(getattr(ann, "max_output_chars", 60_000)))
    batches: list[list[tuple[Path, str, int]]] = []
    oversized: list[tuple[Path, str, int]] = []
    current: list[tuple[Path, str, int]] = []
    size = 0
    for item in pending:
        if len(item[1]) > max_batch_chars:
            # 单章比整批上限还大：放进合批只会让「批内只有一个但本身超限」，
            # 每次都被服务端拒绝。单独走章内切块。
            oversized.append(item)
            continue
        # 输出上限按「每个引号约 12 字符游程」粗估，防止一次吐太多被截断
        if current and (
            size + len(item[1]) > max_batch_chars
            or (sum(x[2] for x in current) + item[2]) * 12 > max_out_chars
        ):
            batches.append(current)
            current, size = [], 0
        current.append(item)
        size += len(item[1]) + 1
    if current:
        batches.append(current)

    from collections import Counter

    done = quotes = failed = 0
    calls = 0
    minor: Counter[str] = Counter()
    budget = int(getattr(ann, "max_calls", 0) or 0)

    def _has_budget(n: int) -> bool:
        """调用预算（0=不限）。单次 annotate 内部最多重试 2 次，此处按逻辑调用计。"""
        return not budget or calls + n <= budget

    def _store(batch: list[tuple[Path, str, int]], speakers: list[str], model: str) -> bool:
        """把整批标注按章拆回并落缓存；数量不符则整批作废。"""
        cursor = 0
        staged: list[tuple[str, list[str]]] = []
        for _path, text, n in batch:
            chunk = speakers[cursor : cursor + n]
            if len(chunk) != n:
                return False
            staged.append((text, chunk))
            cursor += n
        for text, chunk in staged:
            cache.put(text, chunk, model=model, roles_sig=sig)
            minor.update(s for s in chunk if s.startswith("其他"))
        return True

    for batch in batches:
        if not _has_budget(1):
            failed += len(batch)
            output_formatter.warning(f"标注调用已达上限 max_calls={budget}，剩余 {len(batch)} 章回退正则匹配")
            continue

        joined = "\n".join(item[1] for item in batch)
        expected = sum(item[2] for item in batch)
        actual = len(quote_spans(joined))
        reason = ""
        if actual != expected:
            # 某章以未闭合引号结尾时，会与下一章的开引号配对，整批偏移一位。
            # 这种批不能送模型（拆回时前半段对、后半段全错），直接降级。
            reason = f"跨章引号配对异常（各章合计 {expected} → 合并后 {actual}）"
        else:
            calls += 1
            try:
                result = annotator.annotate(joined, roles)
                # 预算按**实际 HTTP 请求**记（annotate 内部校验失败会重试），
                # 否则 max_calls=3 时可能真的发出 9 次请求。
                calls += max(0, int(getattr(result, "runs", 1) or 1) - 1)
                ok = _store(batch, result.speakers, result.model)
                if ok:
                    done += len(batch)
                    quotes += len(result.speakers)
                    continue
                reason = "标注结果与章节引号数不符"
            except AnnotationError as e:
                reason = str(e)
            except Exception as e:  # noqa: BLE001
                reason = f"{type(e).__name__}: {e}"

        # 整批不合格 → 拆成单章重试，避免一章拖累整批（每章 1 次调用）
        if len(batch) > 1 and _has_budget(len(batch)):
            output_formatter.warning(
                f"整批标注失败（{len(batch)} 章）：{reason}；拆成单章重试"
            )
            for item in batch:
                if not _has_budget(1):
                    failed += 1
                    continue
                calls += 1
                try:
                    one = annotator.annotate(item[1], roles)
                    calls += max(0, int(getattr(one, "runs", 1) or 1) - 1)
                except Exception:  # noqa: BLE001
                    failed += 1
                    continue
                if _store([item], one.speakers, one.model):
                    done += 1
                    quotes += len(one.speakers)
                else:
                    failed += 1
            continue

        failed += len(batch)
        output_formatter.warning(f"标注失败（{len(batch)} 章），这些章回退正则匹配: {reason}")

    # 超长章节：按段落切块 → 逐块标注 → 拼接成整章再落缓存。
    # 段落边界保证引号不跨块（引号不会跨段成对），因此拼接后顺序与整章一致。
    for path, text, n in oversized:
        if not _has_budget(1):
            failed += 1
            output_formatter.warning(f"标注调用已达上限 max_calls={budget}，{path.name} 回退正则匹配")
            continue
        chunks = _split_oversized(text, max_batch_chars)
        collected: list[str] = []
        ok = True
        model = ""
        for chunk in chunks:
            if not _has_budget(1):
                ok = False
                break
            calls += 1
            try:
                part = annotator.annotate(chunk, roles)
                calls += max(0, int(getattr(part, "runs", 1) or 1) - 1)
            except Exception as exc:  # noqa: BLE001
                output_formatter.warning(f"超长章节切块标注失败（{path.name}）: {exc}")
                ok = False
                break
            model = part.model
            if len(part.speakers) != len(quote_spans(chunk)):
                ok = False
                break
            collected.extend(part.speakers)
        if not ok or len(collected) != n:
            failed += 1
            output_formatter.warning(f"超长章节标注失败（{path.name}，{len(text)} 字），回退正则匹配")
            continue
        cache.put(text, collected, model=model, roles_sig=sig)
        minor.update(x for x in collected if x.startswith("其他"))
        done += 1
        quotes += len(collected)
        output_formatter.info(
            f"超长章节 {path.name}（{len(text)} 字）切成 {len(chunks)} 块完成标注"
        )

    output_formatter.info(
        f"AI 标注完成：新标注 {done} 章 / 缓存命中 {hit} 章 / 失败 {failed} 章，"
        f"共 {quotes} 个引号（{len(batches)} 批 / {calls} 次调用）"
    )
    if minor:
        top = "、".join(f"{name}({cnt})" for name, cnt in minor.most_common(8))
        output_formatter.info(
            f"未入表角色 {len(minor)} 个，已用 default 音色：{top}"
            f"{' …' if len(minor) > 8 else ''}；建议补进 cast.yml"
        )
    if done:
        output_formatter.info(f"标注缓存目录: {directory}")


def _write_annotated_previews(
    options: "ConvertOptions",
    output_dir: Path,
    config: "PPCConfig",
    output_formatter: "OutputFormatter",
) -> int:
    """把缓存里的标注渲染成 ``<章节>.annotated.txt``（不合成音频）。

    用途：人工校对「这段到底谁读」——可疑的换人、`?` 兜底、其他:* 命名，
    都能一眼看出来；改完 cast 或用编辑器修正后重跑即可复用。
    """
    from ...text.ai_annotate import AnnotationCache, roles_signature
    from ...text.annotate import render_markers
    from ...text.cast import Cast

    cast_path = str(getattr(config.tts, "cast_file", "") or "")
    if not cast_path:
        output_formatter.warning("标注预览跳过：未指定 cast 文件")
        return 0
    try:
        cast = Cast.load(cast_path)
    except Exception as e:  # noqa: BLE001
        output_formatter.warning(f"标注预览跳过：cast 加载失败 {e}")
        return 0

    ann = config.tts.annotate
    directory = str(getattr(ann, "cache_dir", "") or "") or str(
        Path.home() / ".cache" / "ppc" / "annotate"
    )
    cache = AnnotationCache(directory)
    sig = roles_signature(cast.role_table())
    policy = str(getattr(ann, "marker_policy", "all") or "all")

    if options.one:
        files = [options.input]
    else:
        pattern = "**/*.txt" if options.recursive else "*.txt"
        files = sorted(options.input.glob(pattern))

    from collections import Counter

    written = 0
    minor: Counter[str] = Counter()
    for path in files:
        try:
            text = _read_text(path)
        except OSError:
            continue
        speakers = cache.get(text, sig)
        if not speakers:
            continue
        target = output_dir / f"{path.stem}.annotated.txt"
        target.write_text(render_markers(text, speakers, policy=policy), encoding="utf-8")
        minor.update(s for s in speakers if s.startswith("其他"))
        written += 1

    output_formatter.info(f"已输出 {written} 份标注预览（marker_policy={policy}），未合成音频")
    if minor:
        top = "、".join(f"{name}({cnt})" for name, cnt in minor.most_common(8))
        output_formatter.info(
            f"未入表角色 {len(minor)} 个，已用 default 音色：{top}"
            f"{' …' if len(minor) > 8 else ''}；建议补进 cast.yml"
        )
    return written


def _build_retry_policy(config: PPCConfig):
    return create_tts_retry_policy(
        max_retries=config.reliability.tts_retry.max_retries,
        base_delay=config.reliability.tts_retry.base_delay,
        max_delay=config.reliability.tts_retry.max_delay,
        exponential_base=config.reliability.tts_retry.exponential_base,
        jitter=config.reliability.tts_retry.jitter,
    )


def _count_segments(input_file: Path, tts_config) -> tuple[int, str | None]:
    """预分段以获取段数（让 --one 进度显示总段数而非 1）。

    返回 ``(段数, 已读文本)``：文本一并交给执行路径复用，避免第二次全量读
    （12.x 单遍 IO）。
    """
    from ...text.segmenter import TextSegmenter
    from ...utils.files import read_text_auto

    try:
        _text, _enc = read_text_auto(input_file)
    except (OSError, UnicodeDecodeError):
        _text = ""
    if _text and _text.strip():
        if len(_text) <= tts_config.max_segment_length:
            _segments = [_text]
        else:
            _segmenter = TextSegmenter.from_config(tts_config)
            _segments = _segmenter.split(_text, tts_config.max_segment_length) or [_text]
    else:
        _segments = []
    return len(_segments), (_text or None)


def _log_convert_details(options: ConvertOptions, output_dir: Path, config: PPCConfig, output_formatter: OutputFormatter) -> None:
    output_formatter.detail(f"输入: {options.input}")
    output_formatter.detail(f"输出目录: {output_dir}")
    output_formatter.detail(f"语音: {config.tts.voice}")
    output_formatter.detail(f"并发数: {config.tts.concurrency}")
    output_formatter.detail(f"音频速度: {config.tts.rate}")
    if options.one:
        output_formatter.detail("模式: --one（单文件无限重试，无超时）")
    if options.timeout_mode is not None:
        output_formatter.detail(f"超时模式: {options.timeout_mode}")
    if options.timeout is not None:
        output_formatter.detail(f"固定超时: {options.timeout}s")
    if options.timeout_multiplier is not None:
        output_formatter.detail(f"超时倍率: {options.timeout_multiplier}x")
    if config.tts.ramp_up_enabled:
        output_formatter.detail(
            f"并发预热: {config.tts.ramp_up_duration:.0f}s 内从 1 逐步增加到{config.tts.concurrency} 并发"
        )
    if options.resume:
        output_formatter.detail("断点续传: 已启用")
    if options.recursive:
        output_formatter.detail("递归模式: 启用")
    if config.tts.merge_cut.enabled:
        mc = config.tts.merge_cut
        output_formatter.detail(
            f"合并截断: 开启（≤{mc.max_merge_pieces} 段 / ≤{mc.max_merge_bytes}B 合并后按字幕时间切分）"
        )
        if mc.subtitle_output:
            output_formatter.detail("字幕输出: 开启 (.srt)")
    if config.tts.llm_assign.enabled:
        output_formatter.detail(f"LLM 分配发声人: {config.tts.llm_assign.model}")


def _validate_convert(
    options: ConvertOptions,
) -> tuple[Path, PPCConfig, OutputFormatter]:
    # C3（P1）：先把并发数拦在 CLI 边界，越界立即失败（不做静默钳制）。
    # 触发条件：``ppc convert ... -c 0``（或负数）。TTSConfig 未开启
    # validate_assignment，下面的 ``config.tts.concurrency = ...`` 赋值可绕过
    # pydantic 的 ge=1；0 一路传到 asyncio.Semaphore(0)，日志先打印
    # "已启动 0 个工作协程" 然后永久挂起（无报错、无退出码）。
    # 后果：脚本化批量转换静默卡死；Web 侧因 max(1, ...) 钳制而不受影响，
    # 导致 CLI/Web 行为不一致。这里显式报错并给出参数错误退出码 2。
    if options.concurrency is not None and options.concurrency < 1:
        raise CLIError(
            E.E_BUSINESS,
            f"无效的并发数: {options.concurrency}，合法范围为 >= 1 的整数",
            hint="例如 -c 1 或 -c 8；并发数太小会拖慢转换，太大可能触发服务端限流",
            exit_code=2,
        )

    output_dir: Path
    if options.one:
        if not options.input.exists():
            raise CLIError(
                E.E_INPUT_NOT_FOUND,
                f"输入文件不存在: {options.input}",
                hint="--one 模式下 input 必须为已存在的单 .txt 文件",
            )
        if options.input.is_dir():
            raise CLIError(
                E.E_INPUT_NOT_FOUND,
                f"--one 模式下 input 必须是文件,不能是目录: {options.input}",
                hint="移除 --one 进入批量模式,或提供单文件路径",
            )
        if options.input.suffix.lower() != ".txt":
            raise CLIError(
                E.E_INPUT_NOT_FOUND,
                f"--one 模式下 input 必须是 .txt 文件: {options.input}",
                hint="提供 .txt 文件路径",
            )
        if options.output is None:
            options.output = options.input.parent
        options.output.mkdir(parents=True, exist_ok=True)
        output_dir = options.output
    else:
        if options.output is None:
            raise CLIError(
                E.E_INPUT_NOT_FOUND,
                "批量模式必须指定输出目录",
                hint="用法: ppc convert <input_dir> <output_dir>  或  ppc convert <file> --one",
            )
        output_dir = options.output

    output_formatter = get_output()

    config_manager = ConfigManager()

    # balanced 预设直接取管理器共享配置：必须深拷贝，避免 CLI 覆盖
    # （voice/concurrency/rate/...）污染全局配置；其余预设每次构建全新实例。
    if options.preset == "balanced":
        config = copy.deepcopy(config_manager.get_config())
    else:
        config = get_preset(options.preset)

    # 未知 preset 警告：get_preset 对未知名称静默回退到 balanced
    if options.preset != "balanced":
        from ...config import get_preset_names
        if options.preset not in get_preset_names():
            output_formatter.warning(f"未知预设 '{options.preset}'，已回退到 balanced 预设")

    if options.voice is not None:
        config.tts.voice = options.voice
    if options.concurrency is not None:
        config.tts.concurrency = options.concurrency
    if options.rate is not None:
        config.tts.rate = _normalize_rate(options.rate)
    # `--annotate-*` 只在显式给了 `--annotate` 时生效是错的：配置文件里已开
    # annotate 的用户想临时换 key/模型，会被静默忽略（还按旧 key 计费）。
    # 现在只要给了任一 annotate 覆盖参数，就一并应用到配置。
    annotate_overrides = {
        "model": getattr(options, "annotate_model", None),
        "base_url": getattr(options, "annotate_base_url", None),
        "api_key": getattr(options, "annotate_api_key", None),
    }
    if getattr(options, "annotate", False) or any(annotate_overrides.values()):
        config.tts.annotate.enabled = True
        for field, value in annotate_overrides.items():
            if value:
                setattr(config.tts.annotate, field, value)
    if getattr(options, "strip_attribution", False):
        config.tts.strip_attribution.enabled = True
    if getattr(options, "keep_segment_cache", False):
        config.tts.merge_cut.cleanup_segment_cache = False

    if getattr(options, "timeout_mode", None):
        allowed = {"fixed", "auto", "adaptive", "synthetic"}
        if options.timeout_mode not in allowed:
            raise CLIError(
                E.E_BUSINESS,
                f"无效的 timeout-mode: '{options.timeout_mode}'，可选 {sorted(allowed)}",
                hint="拼写错误会被超时计算器当成 auto，导致超时策略与预期不符",
            )

    if options.pitch is not None:
        config.tts.pitch = _normalize_pitch(options.pitch)
    if options.timeout_multiplier is not None:
        config.tts.timeout_multiplier = options.timeout_multiplier
    if options.timeout_mode is not None:
        config.tts.timeout_mode = options.timeout_mode
    if options.timeout is not None:
        config.tts.timeout = options.timeout
    if options.ramp_up is not None:
        config.tts.ramp_up_enabled = True
        config.tts.ramp_up_duration = options.ramp_up
    if options.cast is not None:
        config.tts.cast_file = options.cast

    # ---- 多角色合并优化 ------------------------------------------------
    if options.merge_cut or options.subtitles:
        config.tts.merge_cut.enabled = True
    if options.subtitles:
        config.tts.merge_cut.subtitle_output = True
    if options.merge_max_chars is not None:
        config.tts.merge_cut.max_merge_chars = options.merge_max_chars
    if options.merge_max_bytes is not None:
        config.tts.merge_cut.max_merge_bytes = options.merge_max_bytes
    if options.merge_max_pieces is not None:
        config.tts.merge_cut.max_merge_pieces = options.merge_max_pieces

    # ---- LLM 发声人分配 ------------------------------------------------
    if options.llm_assign:
        config.tts.llm_assign.enabled = True
    if options.llm_model is not None:
        config.tts.llm_assign.model = options.llm_model
    if options.llm_base_url is not None:
        config.tts.llm_assign.base_url = options.llm_base_url
    if options.llm_api_key is not None:
        config.tts.llm_assign.api_key = options.llm_api_key
    if options.main_roles is not None:
        config.tts.llm_assign.main_roles = _parse_role_map(options.main_roles)
    if options.candidate_voices is not None:
        config.tts.llm_assign.candidate_voices = [
            v.strip() for v in options.candidate_voices.split(",") if v.strip()
        ]
    if options.save_cast is not None:
        config.tts.llm_assign.persist_cast = options.save_cast

    _log_convert_details(options, output_dir, config, output_formatter)

    if not options.one and not options.input.exists():
        raise CLIError(
            E.E_INPUT_NOT_FOUND,
            f"输入目录不存在: {options.input}",
            hint="请检查路径是否正确,或使用绝对路径",
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    return output_dir, config, output_formatter


def _run_convert_one(
    options: ConvertOptions,
    output_dir: Path,
    config: PPCConfig,
    output_formatter: OutputFormatter,
) -> None:
    output_file = output_dir / (options.input.stem + ".mp3")

    total_segments, prefetched_text = _count_segments(options.input, config.tts)
    if total_segments == 0:
        raise CLIError(
            E.E_INPUT_EMPTY,
            f"输入文件内容为空，无法转换: {options.input}",
            hint="请提供非空 .txt 文件，或检查文件编码",
        )
    output_formatter.info(f"分段数: {total_segments}")

    # 单文件模式直接走 execute_one
    async def run_convert_one():
        retry_policy = _build_retry_policy(config)

        progress_handler = SimpleProgressHandler(total_tasks=1, formatter=output_formatter)

        async with TTSExecutor(
            config, retry_policy, tts_backend=options.tts_backend, tts_proxy=options.tts_proxy
        ) as executor:
            progress_handler.start()
            try:
                ok = await executor.execute_one(
                    options.input, output_file, progress_handler=progress_handler, text=prefetched_text
                )
            except KeyboardInterrupt:
                output_formatter.warning_panel("用户中断", title="中断")
                return False
            finally:
                progress_handler.stop()
            return ok

    def _render_one_result(ok: bool):
        if ok:
            output_formatter.success_panel(
                f"已生成: {output_file}",
                title="完成",
                details={"文件": str(output_file)},
            )
        return ok

    run_cli_async(
        run_convert_one(),
        failure_message="--one 模式未成功",
        failure_exit_code=1,
        catch_io_errors=True,
        timeout_hint="可尝试 --timeout-multiplier 增大超时时间",
        on_result=_render_one_result,
    )


def _print_failed_tasks(output_formatter: OutputFormatter, progress_handler: SimpleProgressHandler) -> None:
    error_tasks = [
        (info.name, info.error)
        for info in progress_handler.task_infos.values()
        if info.status == "failed" and info.error
    ][:5]

    if error_tasks and output_formatter.mode == "human":
        error_rows = [
            [task_name, error_msg[:50] + "..." if len(error_msg) > 50 else error_msg]
            for task_name, error_msg in error_tasks
        ]
        output_formatter.print_table(
            ["任务名", "错误信息"],
            error_rows,
            title="失败任务示例",
        )


def _collect_pending_files(
    options: ConvertOptions,
    output_dir: Path,
    output_formatter: OutputFormatter,
) -> list[Path] | None:
    # 预扫描文件用于空检查与日志展示；实际任务注册与处理由
    # add_batch_with_progress 内部独立发现（保持单一数据源）
    # 此处的 files_to_process 计数可能略大于实际注册数（断点续传场景），
    # progress_handler.total_tasks 会在 executor 内同步为实际值
    txt_files = sorted(options.input.rglob("*.txt")) if options.recursive else sorted(options.input.glob("*.txt"))
    if not txt_files:
        msg = f"No .txt files found in {options.input}. Nothing to do."
        if options.strict:
            raise CLIError(
                E.E_INPUT_EMPTY,
                msg,
                hint="Remove --strict to allow empty input",
            )
        output_formatter.info(msg)
        return None

    files_to_process = []
    for txt_file in txt_files:
        if options.recursive:
            rel_path = txt_file.relative_to(options.input)
            output_file = output_dir / rel_path.with_suffix(".mp3")
        else:
            output_file = output_dir / txt_file.with_suffix(".mp3").name
        if not output_file.exists() or output_file.stat().st_size == 0:
            files_to_process.append(txt_file)

    if not files_to_process:
        output_formatter.success_panel("所有文件已处理完成，无需转换", title="完成", details={"已处理": len(txt_files)})
        return None

    return files_to_process


def _assemble_book(
    options: ConvertOptions,
    output_dir: Path,
    config: PPCConfig,
    output_formatter: OutputFormatter,
) -> None:
    """12.x W3-1 ``--book``：把本批章节 MP3 装配为带 ID3v2 章节标记的整书。

    零解码帧拼接 + CTOC/CHAP 内嵌——单文件、任意主流播放器可直接跳章。
    """
    from ...audio.mp3_book import BookChapter, BookMeta, merge_chapters_to_book

    txt_files = sorted(options.input.rglob("*.txt")) if options.recursive else sorted(options.input.glob("*.txt"))
    chapters: list[BookChapter] = []
    missing: list[str] = []
    for txt in txt_files:
        if options.recursive:
            mp3 = output_dir / txt.relative_to(options.input).with_suffix(".mp3")
        else:
            mp3 = output_dir / txt.with_suffix(".mp3").name
        if mp3.exists() and mp3.stat().st_size > 0:
            chapters.append(BookChapter(mp3, title=txt.stem))
        else:
            missing.append(txt.stem)

    if not chapters:
        output_formatter.warning("--book: 没有已完成的章节音频，跳过整书装配")
        return

    title = options.book_title or options.input.name
    book_path = output_dir / f"{title}.book.mp3"
    meta = BookMeta(
        title=title,
        album=title,
        embed=bool(getattr(config.output, "metadata_embed", True)),
    )
    silence_ms = int(getattr(config.output, "silence_between_chapters_ms", 500))

    result = merge_chapters_to_book(chapters, book_path, meta=meta, silence_ms=silence_ms)
    if not result.success:
        reason = result.degraded_reason or result.error
        output_formatter.warning(f"--book 装配失败（章节参数不一致或不可探测: {reason}），已保留分章产物")
        return

    minutes = result.duration_seconds / 60.0
    details = {"整书": str(book_path), "章节": f"{result.chapter_count} 章（已内嵌播放列表）", "时长": f"{minutes:.1f} 分钟"}
    if missing:
        details["缺失章节"] = f"{len(missing)} 个未生成: " + ", ".join(missing[:3]) + ("…" if len(missing) > 3 else "")
    output_formatter.success_panel(f"整书已生成: {book_path}", title="BOOK", details=details)


def _run_convert_batch(
    options: ConvertOptions,
    output_dir: Path,
    config: PPCConfig,
    output_formatter: OutputFormatter,
) -> None:
    files_to_process = _collect_pending_files(options, output_dir, output_formatter)
    if files_to_process is None:
        return

    output_formatter.info(f"待处理文件数: {len(files_to_process)}")

    async def run_convert():
        retry_policy = _build_retry_policy(config)
        progress_handler = SimpleProgressHandler(len(files_to_process), formatter=output_formatter)

        async with TTSExecutor(
            config, retry_policy, tts_backend=options.tts_backend, tts_proxy=options.tts_proxy
        ) as executor:
            if options.resume:
                ckpt_path = options.checkpoint or output_dir / ".ppc_checkpoint.json"
                executor.enable_checkpoint(ckpt_path)

            executor.set_progress_callback(progress_handler)
            progress_handler.start()

            try:
                await executor.add_batch_with_progress(
                    options.input, output_dir, progress_handler=progress_handler, recursive=options.recursive
                )
            except KeyboardInterrupt:
                output_formatter.warning_panel("用户中断，检查点已保存", title="中断")
                output_formatter.info("使用 --resume 参数可从中断处继续")
                return False
            finally:
                progress_handler.stop()

            stats = progress_handler.get_detailed_stats()

            output_formatter.completion_report(
                stats,
                executor_stats=None,
                title="转换完成报告",
            )

            failed = stats["failed"]
            if failed > 0 and progress_handler.task_infos:
                _print_failed_tasks(output_formatter, progress_handler)

            if options.book:
                try:
                    _assemble_book(options, output_dir, config, output_formatter)
                except Exception as exc:  # 装配失败不影响批量转换退出码
                    output_formatter.warning(f"--book 装配异常: {exc}")

            return failed == 0

    def _warn_interrupt():
        output_formatter.warning("\n用户中断")

    # 长时间合成期间阻止系统休眠（失败时工具内部自动降级为无操作）
    from ...utils.keep_awake import keep_awake_context

    with keep_awake_context():
        run_cli_async(
            run_convert(),
            failure_message="存在失败任务,请查看上方重试/错误明细",
            failure_hint="使用 --verbose 查看详细错误信息,或调整 --timeout-multiplier / 减少并发",
            interrupt_message="用户中断 (Ctrl+C)",
            on_interrupt=_warn_interrupt,
            catch_io_errors=True,
        )


def handle_convert(
    options: ConvertOptions,
) -> None:
    """处理转换命令

    - one=False: 批量模式，处理 input 目录下所有 .txt → output
    - one=True : 单文件模式，input 为单 .txt，output 可缺省（默认 input.parent）
    """
    output_dir, config, output_formatter = _validate_convert(options)

    # LLM 发声人分配（可选）：在构建 TTSEngine 之前把 cast 落好
    _apply_llm_assign(options, output_dir, config, output_formatter)
    # AI 引号级标注（可选）：写章节缓存，引擎按指纹取用
    _apply_annotate(options, output_dir, config, output_formatter)
    if getattr(options, "annotate_only", False):
        _write_annotated_previews(options, output_dir, config, output_formatter)
        return

    if options.one:
        if options.book:
            raise CLIError(E.E_BUSINESS, "--book 适用于目录批量模式（章节→整书），与 --one 互斥")
        _run_convert_one(options, output_dir, config, output_formatter)
        return

    _run_convert_batch(options, output_dir, config, output_formatter)
