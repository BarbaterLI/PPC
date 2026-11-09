"""AI 能力（发声人分配 / 说话人标注）的共享实现。

``ppc ai`` 的 assign / annotate 子命令都调用这里，是 AI 侧的唯一实现层。
单独成模块而不是留在 ``convert`` 里，是为了让 ``ppc ai`` 不必导入 TTS 引擎与
并发执行器（``convert`` 会拖进一大票重型依赖，启动成本高）。

两个不能动的约束
----------------
1. ``apply_llm_assign`` / ``apply_annotate`` 的**函数体内**才 import
   ``LLMSpeakerAssigner`` / ``AIAnnotator``：测试用 monkeypatch 替换的是模块
   属性，提到顶层会让补丁彻底失效。
2. 三个 helper 的参数顺序 ``(options, output_dir, config, output_formatter)``
   被测试按位置调用，不能改成关键字或调整顺序。
"""

from __future__ import annotations

import copy
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from ...config import ConfigManager, get_preset
from ...core.exceptions import CLIError
from ...core.exceptions import ErrorCodes as E

if TYPE_CHECKING:
    from src.cli.output import OutputFormatter
    from src.config import PPCConfig

    class AiTargetSource(Protocol):
        """三个 helper 只需要这三个字段，``ConvertOptions`` 天然满足。"""

        input: Path
        one: bool
        recursive: bool


__all__ = [
    "AiOverrides",
    "AiTargets",
    "apply_ai_overrides",
    "apply_annotate",
    "apply_llm_assign",
    "build_ai_config",
    "parse_role_map",
    "read_text",
    "resolve_ai_target",
    "split_oversized",
    "write_annotated_previews",
]


@dataclass
class AiTargets:
    """AI 命令的输入目标（替代 ``ConvertOptions``，只保留真正用到的字段）。"""

    input: Path
    one: bool = False
    recursive: bool = False


@dataclass
class AiOverrides:
    """CLI 开关 → config 的翻译入参。

    ``None`` / ``False`` 一律表示「不覆盖」，与原来的 flag 语义逐字对应：
    只有显式给了值才写 config，否则沿用配置文件。
    """

    annotate: bool = False
    annotate_model: str | None = None
    annotate_base_url: str | None = None
    annotate_api_key: str | None = None
    annotate_max_calls: int | None = None
    llm_assign: bool = False
    llm_model: str | None = None
    llm_base_url: str | None = None
    llm_api_key: str | None = None
    main_roles: str | None = None
    candidate_voices: str | None = None
    save_cast: str | None = None


def build_ai_config(
    preset: str = "balanced", output_formatter: OutputFormatter | None = None
) -> PPCConfig:
    """按预设构建配置（与 ``convert`` 的历史行为一致）。

    ``balanced`` 取管理器共享配置的深拷贝——CLI 会就地覆写 voice/rate 等字段，
    不拷贝会污染进程内的全局配置；其余预设每次新建实例。
    """
    if preset == "balanced":  # noqa: SIM108 - 两分支各带注释，三元会损失可读性
        config = copy.deepcopy(ConfigManager().get_config())
    else:
        config = get_preset(preset)

    # 未知 preset 警告：get_preset 对未知名称静默回退到 balanced
    if preset != "balanced" and output_formatter is not None:
        from ...config import get_preset_names

        if preset not in get_preset_names():
            output_formatter.warning(f"未知预设 '{preset}'，已回退到 balanced 预设")
    return config


def apply_ai_overrides(config: PPCConfig, overrides: AiOverrides) -> PPCConfig:
    """把 AI 相关开关写进 config（原 ``_validate_convert`` 里的两段合并）。

    注意 annotate 的非对称规则：只要出现任意一个 ``--annotate-*`` 覆盖项，
    即使没写 ``--annotate`` 也会启用标注——否则用户临时换模型/Key 会被静默忽略。
    """
    o = overrides
    annotate_overrides = {
        "model": o.annotate_model,
        "base_url": o.annotate_base_url,
        "api_key": o.annotate_api_key,
    }
    if o.annotate or any(annotate_overrides.values()):
        config.tts.annotate.enabled = True
        for field_name, value in annotate_overrides.items():
            if value:
                setattr(config.tts.annotate, field_name, value)
    if o.annotate_max_calls is not None:
        config.tts.annotate.max_calls = o.annotate_max_calls

    if o.llm_assign:
        config.tts.llm_assign.enabled = True
    if o.llm_model is not None:
        config.tts.llm_assign.model = o.llm_model
    if o.llm_base_url is not None:
        config.tts.llm_assign.base_url = o.llm_base_url
    if o.llm_api_key is not None:
        config.tts.llm_assign.api_key = o.llm_api_key
    if o.main_roles is not None:
        config.tts.llm_assign.main_roles = parse_role_map(o.main_roles)
    if o.candidate_voices is not None:
        config.tts.llm_assign.candidate_voices = [
            v.strip() for v in o.candidate_voices.split(",") if v.strip()
        ]
    if o.save_cast is not None:
        config.tts.llm_assign.persist_cast = o.save_cast
    return config


def resolve_ai_target(
    input_path: Path, output: Path | None, *, one: bool, recursive: bool
) -> tuple[Path, AiTargets]:
    """校验输入并给出工作目录（错误文案对齐 ``convert`` 的既有提示）。"""
    if one:
        if not input_path.exists():
            raise CLIError(
                E.E_INPUT_NOT_FOUND,
                f"输入文件不存在: {input_path}",
                hint="--one 模式下 input 必须为已存在的单 .txt 文件",
            )
        if input_path.is_dir():
            raise CLIError(
                E.E_INPUT_NOT_FOUND,
                f"--one 模式下 input 必须是文件,不能是目录: {input_path}",
                hint="移除 --one 进入批量模式,或提供单文件路径",
            )
        if input_path.suffix.lower() != ".txt":
            raise CLIError(
                E.E_INPUT_NOT_FOUND,
                f"--one 模式下 input 必须是 .txt 文件: {input_path}",
                hint="提供 .txt 文件路径",
            )
        out = output or input_path.parent
    else:
        if input_path.is_file():
            raise CLIError(
                E.E_BUSINESS,
                f"批量模式输入必须是目录: {input_path}",
                hint="单文件请加 --one",
                exit_code=2,
            )
        if not input_path.exists():
            raise CLIError(
                E.E_INPUT_NOT_FOUND,
                f"输入目录不存在: {input_path}",
                hint="请检查路径是否正确",
            )
        out = output or input_path

    out.mkdir(parents=True, exist_ok=True)
    return out, AiTargets(input=input_path, one=one, recursive=recursive)


def parse_role_map(value: str) -> dict[str, str]:
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

def read_text(path: Path) -> str:
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

def apply_llm_assign(
    options: AiTargetSource,
    output_dir: Path,
    config: PPCConfig,
    output_formatter: OutputFormatter,
) -> None:
    """转换前用 LLM 生成 cast 并注入 ``tts.cast_file``。

    失败（未装 openai / 无 key / 网络或额度问题）只警告并降级为现有 cast，
    绝不阻断转换。
    """
    if not getattr(config.tts.llm_assign, "enabled", False):
        return

    from ...text.cast import Cast
    from ...text.llm_assign import LLMSpeakerAssigner, LLMSpeakerAssignError

    texts: list[str] = []
    if options.one:
        try:
            texts.append(read_text(options.input))
        except OSError as e:
            output_formatter.warning(f"LLM 分配跳过：读取输入失败 {e}")
            return
    else:
        pattern = "**/*.txt" if options.recursive else "*.txt"
        for txt in sorted(options.input.glob(pattern)):
            try:
                texts.append(read_text(txt))
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

def split_oversized(text: str, limit: int) -> list[str]:
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

def apply_annotate(
    options: AiTargetSource,
    output_dir: Path,
    config: PPCConfig,
    output_formatter: OutputFormatter,
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
            text = read_text(path)
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
    # token 预算是更靠上的约束：用户给了就以它为准（0=沿用字符上限）
    input_budget = int(getattr(ann, "max_input_tokens", 0) or 0)
    if input_budget > 0:
        from ...text.token_budget import chars_for_tokens

        max_batch_chars = min(max_batch_chars, max(1000, chars_for_tokens(input_budget)))
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
        chunks = split_oversized(text, max_batch_chars)
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

def write_annotated_previews(
    options: AiTargetSource,
    output_dir: Path,
    config: PPCConfig,
    output_formatter: OutputFormatter,
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
            text = read_text(path)
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

