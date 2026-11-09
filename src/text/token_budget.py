"""把「token 预算」翻译成「每批几条、每条多少上下文」。

背景
----
各 AI 功能的批次大小原本是按**字符数**硬编码的（``chunk_chars=6000``、
``max_input_chars=8 万``）。真正受限的是模型的 token 窗口，而且推理模型
（DeepSeek-V4.1-Flash 等）的输出预算还要先被 reasoning tokens 吃掉一截——
实测一次 ``max_tokens=16`` 的调用里 14 个 token 花在推理上，正文只剩 1 个。

于是统一成两个用户旋钮：

* ``max_input_tokens``  —— 单次请求允许塞多少上下文
* ``max_output_tokens`` —— 单次请求允许模型吐多少（含推理）

其余参数（每批条数、每条附带的上下文字数、总批数）全部由这两个值推导，
不再散落在各功能里各写一套。

估算口径
--------
中文按 **1 字 ≈ 1 token**（DeepSeek/GPT 系中文实测 1~1.5 字/token，取保守端），
ASCII 按 4 字符/token。宁高估不低估：批次偏小只是多花几次调用，超窗口是硬失败。
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

__all__ = ["ContextPlan", "chars_for_tokens", "estimate_tokens", "plan_context"]

_CJK_RE = re.compile(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]")
_LATIN_RUN_RE = re.compile(r"[A-Za-z0-9_]+")

#: 单条消息的信封开销（role/content 包装），保守取 8
_MESSAGE_OVERHEAD = 8


def estimate_tokens(text: str) -> int:
    """估算一段文本的 token 数（保守）。"""
    if not text:
        return 0
    cjk = len(_CJK_RE.findall(text))
    latin_chars = sum(len(m) for m in _LATIN_RUN_RE.findall(text))
    # 标点/空白按 2 字符 1 token 粗算
    other = max(0, len(text) - cjk - latin_chars)
    tokens = cjk + math.ceil(latin_chars / 4) + math.ceil(other / 2)
    return tokens + _MESSAGE_OVERHEAD


def chars_for_tokens(tokens: int) -> int:
    """token 预算能装多少中文字（保守取 0.9 字/token 的余量）。"""
    if tokens <= 0:
        return 0
    return int(tokens * 0.9)


@dataclass(frozen=True)
class ContextPlan:
    """由 token 预算推导出的批次方案。"""

    total_items: int
    items_per_call: int
    calls: int
    context_chars: int
    input_tokens_per_call: int
    output_tokens_per_call: int


def plan_context(
    total_items: int,
    *,
    max_input_tokens: int,
    max_output_tokens: int,
    static_prompt_tokens: int = 800,
    item_overhead_tokens: int = 12,
    output_tokens_per_item: int = 40,
    reasoning_reserve_tokens: int = 600,
    min_context_chars: int = 40,
    max_context_chars: int = 300,
    max_items_per_call: int | None = None,
) -> ContextPlan:
    """给定 token 预算与待处理条目数，推导每批条数与每条上下文长度。

    先由**输出预算**定每批能处理多少条（每条至少一行结论 + 推理余量），
    再由**输入预算**除以条数，得到每条可用的上下文字数——这样两个旋钮
    共同决定了「一条候选能看到多少原文」。
    """
    if total_items <= 0:
        return ContextPlan(0, 0, 0, 0, 0, 0)

    usable_output = max(200, max_output_tokens - reasoning_reserve_tokens)
    items_per_call = max(1, usable_output // max(1, output_tokens_per_item))
    # 推理模型的实际输出长度波动很大，单批条数给个硬上限更稳（也控制单次延迟）
    if max_items_per_call is not None:
        items_per_call = min(items_per_call, max(1, max_items_per_call))
    items_per_call = min(items_per_call, total_items)
    calls = math.ceil(total_items / items_per_call)

    avail_input = max(1000, max_input_tokens - static_prompt_tokens)
    per_item_tokens = avail_input / items_per_call - item_overhead_tokens
    context_chars = chars_for_tokens(int(max(0, per_item_tokens)))
    context_chars = max(min_context_chars, min(max_context_chars, context_chars))

    return ContextPlan(
        total_items=total_items,
        items_per_call=items_per_call,
        calls=calls,
        context_chars=context_chars,
        input_tokens_per_call=items_per_call * (context_chars + item_overhead_tokens),
        output_tokens_per_call=items_per_call * output_tokens_per_item,
    )
