"""``ppc ai`` 命令组：把 AI 能力从 ``convert`` 中独立出来。

为什么单独成组
--------------
``convert`` 的职责是「读章节 → 合成 → 合并」，而发声人分配与引号级标注都是
**在合成之前**、会调外部模型、会失败降级的独立工序。挂成 convert 的开关导致
三个问题：不能单独重跑、失败降级藏在合成日志里、以及 `--annotate-*` 这类
覆盖参数与合成参数混在同一张选项表里。

命令与旧开关的对应关系（旧开关已移除）::

    ppc convert --llm-assign   →  ppc ai assign
    ppc convert --annotate     →  ppc ai annotate
    ppc convert --annotate-only →  ppc ai annotate --preview   （也是默认行为）

``discover`` 目前是纯本地统计（不调模型）；按项目约定它仍留在这个组里，
若将来 cast 相关能力独立成 ``ppc cast``，它会随之迁走。
"""

from __future__ import annotations

from pathlib import Path

import typer

from ...core.exceptions import CLIError
from ...core.exceptions import ErrorCodes as E
from ..output import get_output

ai_app = typer.Typer(
    name="ai",
    help="AI 辅助：发声人分配（assign）/ 说话人标注（annotate）/ 自检（doctor）/ 候选发现（discover）",
    add_completion=False,
    rich_markup_mode="rich",
)

_DEFAULT_CAST_NAME = "cast.yml"


@ai_app.command("assign")
def ai_assign(
    input: Path = typer.Argument(..., help="输入 .txt 文件（配合 --one）或输入目录"),
    output: Path | None = typer.Option(
        None, "--output", "-o", help="工作目录（缺省：--one 时输入同目录，否则输入目录本身）"
    ),
    one: bool = typer.Option(False, "--one", help="单文件模式"),
    recursive: bool = typer.Option(False, "--recursive", "-R", help="递归处理子目录"),
    preset: str = typer.Option("balanced", "--preset", "-p", help="配置预设"),
    cast: str | None = typer.Option(
        None, "--cast", help="作为合并基线的现有 cast.yml（只读，不会被写）"
    ),
    model: str | None = typer.Option(None, "--model", help="LLM 模型名（默认读配置 tts.llm_assign.model）"),
    base_url: str | None = typer.Option(None, "--base-url", help="OpenAI 兼容端点"),
    api_key: str | None = typer.Option(None, "--api-key", help="API Key（默认读 PPC_LLM_API_KEY）"),
    main_roles: str | None = typer.Option(
        None, "--main-roles", help="主要角色→音色：JSON 串 / 文件 / '角色=音色,角色=音色'"
    ),
    candidate_voices: str | None = typer.Option(
        None, "--candidate-voices", help="候选音色池，逗号分隔（LLM 只能从中选）"
    ),
    save_cast: Path | None = typer.Option(
        None, "--save-cast", help=f"落盘路径（缺省 <工作目录>/.cache/{_DEFAULT_CAST_NAME}）"
    ),
    force: bool = typer.Option(False, "--force", "-f", help="允许覆盖 --save-cast 指定的已存在文件"),
) -> None:
    """用 LLM 自动分配发声人并生成 cast 文件（取代已移除的 convert --llm-assign）。

    默认写到 ``<工作目录>/.cache/cast.yml``：``.cache/`` 是运行时产物目录，
    重复运行覆盖它不会伤到用户的配置；而 ``--save-cast`` 指向的文件可能是手写
    的 cast.yml，没有 ``--force`` 一律拒绝覆盖。
    """
    from .ai_common import (
        AiOverrides,
        apply_ai_overrides,
        apply_llm_assign,
        build_ai_config,
        resolve_ai_target,
    )

    fmt = get_output()
    out_dir, targets = resolve_ai_target(input, output, one=one, recursive=recursive)

    target = Path(save_cast) if save_cast is not None else out_dir / ".cache" / _DEFAULT_CAST_NAME
    if save_cast is not None and target.exists() and not force:
        raise CLIError(
            E.E_BUSINESS,
            f"目标文件已存在，拒绝覆盖: {target}",
            hint="确认要覆盖请加 --force；或换一个 --save-cast 路径",
            exit_code=2,
        )
    target.parent.mkdir(parents=True, exist_ok=True)

    config = build_ai_config(preset, fmt)
    if cast:
        # 只作为合并基线：LLMSpeakerAssigner 会把它没观察到的角色继承下来
        config.tts.cast_file = cast

    apply_ai_overrides(
        config,
        AiOverrides(
            llm_assign=True,
            llm_model=model,
            llm_base_url=base_url,
            llm_api_key=api_key,
            main_roles=main_roles,
            candidate_voices=candidate_voices,
            save_cast=str(target),
        ),
    )

    apply_llm_assign(targets, out_dir, config, fmt)
    if not target.exists():
        raise CLIError(
            E.E_BUSINESS,
            "LLM 分配失败，未生成 cast 文件",
            hint="检查 --api-key / --base-url 与网络；告警详情可用 --verbose 查看",
        )
    fmt.info(f"下一步：ppc convert {input} <output> --cast {target}")


@ai_app.command("annotate")
def ai_annotate(
    input: Path = typer.Argument(..., help="输入 .txt 文件（配合 --one）或输入目录"),
    output: Path | None = typer.Option(None, "--output", "-o", help="工作目录（预览文件写在这里）"),
    one: bool = typer.Option(False, "--one", help="单文件模式"),
    recursive: bool = typer.Option(False, "--recursive", "-R", help="递归处理子目录"),
    preset: str = typer.Option("balanced", "--preset", "-p", help="配置预设"),
    cast: str | None = typer.Option(None, "--cast", help="cast.yml 路径（覆盖配置 tts.cast_file）"),
    model: str | None = typer.Option(None, "--model", help="标注用模型名（默认读配置 tts.annotate.model）"),
    base_url: str | None = typer.Option(None, "--base-url", help="标注端点（OpenAI 兼容）"),
    api_key: str | None = typer.Option(None, "--api-key", help="标注 API Key（默认读 PPC_LLM_API_KEY）"),
    max_calls: int | None = typer.Option(
        None, "--max-calls", min=0, help="本次 LLM 请求预算上限（0=不限），覆盖配置 tts.annotate.max_calls"
    ),
    preview: bool = typer.Option(
        True, "--preview/--no-preview", help="输出 <章节>.annotated.txt 供人工校对；关闭则只预热缓存"
    ),
) -> None:
    """用 LLM 逐引号标注说话人（取代已移除的 convert --annotate / --annotate-only）。

    结果按「章节正文 + 角色表」指纹缓存；失败整章回退正则匹配器，不阻断后续合成。
    """
    from .ai_common import (
        AiOverrides,
        apply_ai_overrides,
        apply_annotate,
        build_ai_config,
        resolve_ai_target,
        write_annotated_previews,
    )

    fmt = get_output()
    out_dir, targets = resolve_ai_target(input, output, one=one, recursive=recursive)

    config = build_ai_config(preset, fmt)
    if cast:
        config.tts.cast_file = cast
    if not str(getattr(config.tts, "cast_file", "") or ""):
        raise CLIError(
            E.E_BUSINESS,
            "未指定 cast 文件，无法确定候选角色",
            hint="加 --cast cast.yml，或先运行 ppc ai assign 生成",
            exit_code=2,
        )

    apply_ai_overrides(
        config,
        AiOverrides(
            annotate=True,
            annotate_model=model,
            annotate_base_url=base_url,
            annotate_api_key=api_key,
            annotate_max_calls=max_calls,
        ),
    )

    apply_annotate(targets, out_dir, config, fmt)
    if preview:
        write_annotated_previews(targets, out_dir, config, fmt)


@ai_app.command("doctor")
def ai_doctor(
    network: bool = typer.Option(False, "--network", help="额外发一次最小请求验证端点连通性"),
) -> None:
    """检查 AI 功能的可用性：依赖、Key、端点、缓存目录。

    默认**不联网**：只做本地检查，避免为了看配置而先付一次请求。
    """
    fmt = get_output()
    from .ai_common import build_ai_config

    config = build_ai_config("balanced", fmt)
    assign_cfg = config.tts.llm_assign
    ann_cfg = config.tts.annotate

    try:
        import openai  # noqa: F401

        fmt.info("openai 依赖: 已安装")
    except ImportError:
        fmt.warning("openai 依赖: 未安装（pip install openai），assign / annotate 均不可用")

    rows = [
        ("assign", assign_cfg.base_url, assign_cfg.model, assign_cfg.api_key),
        ("annotate", ann_cfg.base_url, ann_cfg.model, ann_cfg.api_key),
    ]
    for name, base_url, model, api_key in rows:
        key = _mask_key(str(api_key or ""))
        # 未配置 api_key 时运行期还会回退到环境变量，这里要如实说明
        if not api_key:
            import os

            env = os.environ.get("PPC_LLM_API_KEY") or os.environ.get("OPENAI_API_KEY") or ""
            key = f"（环境变量 {_mask_key(env)}）" if env else "缺失"
        fmt.info(f"{name}: model={model or '-'} base_url={base_url or '-'} key={key}")

    cache_dir = str(getattr(ann_cfg, "cache_dir", "") or "")
    if not cache_dir:
        from pathlib import Path as _Path

        cache_dir = str(_Path.home() / ".cache" / "ppc" / "annotate")
    try:
        from pathlib import Path as _Path

        _Path(cache_dir).mkdir(parents=True, exist_ok=True)
        probe = _Path(cache_dir) / ".write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        fmt.info(f"标注缓存目录: {cache_dir}（可写）")
    except OSError as exc:
        fmt.warning(f"标注缓存目录不可写: {cache_dir}（{exc}）")

    if network:
        _probe_endpoint("assign", assign_cfg, fmt)
        _probe_endpoint("annotate", ann_cfg, fmt)


def _mask_key(key: str) -> str:
    """Key 只显示前 4 位，避免 doctor 把密钥打进日志。"""
    if not key:
        return ""
    return key[:4] + "…" if len(key) > 4 else "…"


def _probe_endpoint(name: str, cfg: object, fmt: object) -> None:
    """发一次最小 chat 请求验证端点；失败只告警，不影响退出码。"""
    try:
        from openai import OpenAI
    except ImportError:
        fmt.warning(f"{name} 连通性: 跳过（未安装 openai）")  # type: ignore[attr-defined]
        return

    import os

    api_key = (
        str(getattr(cfg, "api_key", "") or "")
        or os.environ.get("PPC_LLM_API_KEY", "")
        or os.environ.get("OPENAI_API_KEY", "")
    )
    if not api_key:
        fmt.warning(f"{name} 连通性: 跳过（缺 API Key）")  # type: ignore[attr-defined]
        return

    client = OpenAI(
        api_key=api_key,
        base_url=str(getattr(cfg, "base_url", "") or "") or None,
        timeout=float(getattr(cfg, "timeout", 30.0) or 30.0),
    )
    try:
        client.chat.completions.create(
            model=str(getattr(cfg, "model", "")),
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=1,
        )
        fmt.info(f"{name} 连通性: OK")  # type: ignore[attr-defined]
    except Exception as exc:  # noqa: BLE001 - 探活不应因任何异常中断
        fmt.warning(f"{name} 连通性: 失败（{exc}）")  # type: ignore[attr-defined]


def _clip(text: str, limit: int) -> str:
    """表格里只显示上下文开头，完整内容留在草案与 JSON 里。"""
    return text if len(text) <= limit else text[:limit] + "…"


@ai_app.command("discover")
def ai_discover(
    input: Path = typer.Argument(..., help="输入 .txt 文件（配合 --one）或章节目录"),
    output: Path | None = typer.Option(
        None, "--output", "-o", help="工作目录（草案写到 <工作目录>/.cache/cast.draft.yml）"
    ),
    one: bool = typer.Option(False, "--one", help="单文件模式"),
    recursive: bool = typer.Option(False, "--recursive", "-R", help="递归处理子目录"),
    min_count: int = typer.Option(1, "--min-count", min=1, help="候选最少出现次数"),
    min_chapters: int = typer.Option(1, "--min-chapters", min=1, help="候选最少跨章节数"),
    top: int = typer.Option(40, "--top", min=1, max=2000, help="草案里列出多少个候选"),
    max_input_tokens: int = typer.Option(
        60000, "--max-input-tokens", min=1000, help="单次请求的上下文 token 预算（决定每条候选能带多少原文）"
    ),
    max_output_tokens: int = typer.Option(
        8000, "--max-output-tokens", min=256, help="单次请求的输出 token 预算（决定每批处理多少条）"
    ),
    context_chars: int = typer.Option(
        0, "--context-chars", min=0, max=400, help="显式覆盖上下文字数；0=由 token 预算推导"
    ),
) -> None:
    """从正文里宽松召回「可能的角色名」，生成 cast 草案（纯本地，不调模型）。

    只是**候选**：名字与音色仍需人工确认。召回只认「紧邻言说/动作动词」或
    「被点名」的名字，纯叙述里出现的临时角色（医生、店员）不会进入候选——
    它们通常只有一两句台词，交给兜底音色即可。
    """
    from ...text.cast_discover import discover_candidates, render_draft
    from ...text.token_budget import plan_context
    from .ai_common import read_text, resolve_ai_target

    fmt = get_output()
    out_dir, targets = resolve_ai_target(input, output, one=one, recursive=recursive)

    if targets.one:
        files = [targets.input]
    else:
        pattern = "**/*.txt" if targets.recursive else "*.txt"
        files = sorted(targets.input.glob(pattern))
    texts = []
    for path in files:
        try:
            texts.append(read_text(path))
        except OSError:
            continue
    if not texts:
        raise CLIError(
            E.E_INPUT_EMPTY,
            f"没有可分析的文本: {targets.input}",
            hint="确认目录下存在 .txt 章节文件，或改用 --one 指定单文件",
        )

    # 先按上限抓上下文，再按预算裁剪：候选数要等扫描完才知道，
    # 而上下文是扫描时顺手取的，两步比扫两遍便宜。
    candidates = discover_candidates(
        texts, min_count=min_count, min_chapters=min_chapters, context_chars=300
    )
    if not candidates:
        fmt.warning("未召回任何候选：确认输入是小说正文（需要「名字+言说动词」的对白结构）")
        return

    target = out_dir / ".cache" / "cast.draft.yml"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_draft(candidates, limit=top), encoding="utf-8")

    from dataclasses import replace

    frequent = sum(1 for c in candidates if c.count >= 3 and c.chapters >= 2)
    fmt.info(
        f"候选 {len(candidates)} 个（{len(texts)} 章；出现 ≥3 次且跨 ≥2 章：{frequent} 个）"
    )
    # 批次按真正要喂模型的那一批算（Tier1），不是全部候选
    to_model = [c for c in candidates if c.count >= 3 and c.chapters >= 2] or candidates

    if context_chars > 0:
        width = context_chars
        budget_note = f"上下文字数由 --context-chars 指定：{width}"
    else:
        plan = plan_context(
            len(to_model),
            max_input_tokens=max_input_tokens,
            max_output_tokens=max_output_tokens,
        )
        width = plan.context_chars
        budget_note = (
            f"按 token 预算推导：每条上下文 {width} 字；每批 {plan.items_per_call} 条、"
            f"共 {plan.calls} 批（≈{plan.input_tokens_per_call // 1000}k 输入 tokens/批）"
        )
    candidates = [replace(c, context=c.context[:width]) for c in candidates]
    fmt.info(budget_note)
    # detail 只在 --verbose 下渲染，候选名单必须默认可见——它是本命令的全部产出
    shown = candidates[: min(top, 30)]
    fmt.print_table(
        ["候选", "次数", "章", "上下文"],
        [[c.name, str(c.count), str(c.chapters), _clip(c.context, 40)] for c in shown],
        title=f"候选角色（前 {len(shown)} / 共 {len(candidates)}）",
        json_data=[
            {"name": c.name, "count": c.count, "chapters": c.chapters, "context": c.context}
            for c in shown
        ],
    )
    fmt.info(f"草案已写入: {target}")
    fmt.info("下一步：把要用的角色填上音色后另存为 cast.yml，再跑 ppc ai annotate")
