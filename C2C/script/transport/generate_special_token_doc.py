"""Generate a pinned special-token inventory and alignment note."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer


SOURCE = "Qwen/Qwen3-8B"
SOURCE_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
TARGET = "mistralai/Mistral-Nemo-Instruct-2407"
TARGET_REVISION = "04d8a90549d23fc6bd7f642064003592df51e9b3"
DEEPSEEK = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
DEEPSEEK_REVISION = "6a6f4aa4197940add57724a7707d069478df56b1"


def added_tokens(tokenizer: Any) -> dict[int, Any]:
    return {
        int(token_id): token
        for token_id, token in tokenizer.backend_tokenizer.get_added_tokens_decoder().items()
    }


def markdown_token(value: str) -> str:
    escaped = value.replace("|", "\\|")
    return f"`{escaped}`"


def configured_roles(tokenizer: Any, token_id: int) -> str:
    roles = []
    for role in ("bos", "eos", "pad", "unk"):
        if getattr(tokenizer, f"{role}_token_id", None) == token_id:
            roles.append(role.upper())
    if token_id in getattr(tokenizer, "additional_special_tokens_ids", []):
        roles.append("additional_special")
    return ", ".join(roles) or "—"


QWEN_HANDLING = {
    151643: ("PAD / end-of-text", "Mistral `<pad>` 10（需显式配置）"),
    151644: ("chat message start", "模板级序列；无单 token 对应"),
    151645: ("message end / EOS", "Mistral `</s>` 2"),
    151646: ("object reference start", "无对应；排除"),
    151647: ("object reference end", "无对应；排除"),
    151648: ("box start", "无对应；排除"),
    151649: ("box end", "无对应；排除"),
    151650: ("quad start", "无对应；排除"),
    151651: ("quad end", "无对应；排除"),
    151652: ("vision start", "无对应；排除"),
    151653: ("vision end", "无对应；排除"),
    151654: ("vision padding", "无对应；排除"),
    151655: ("image padding", "无对应；排除"),
    151656: ("video padding", "无对应；排除"),
    151657: ("tool call start", "Mistral `[TOOL_CALLS]` 9（近似）"),
    151658: ("tool call end", "无单 token 对应；模板级序列"),
    151659: ("FIM prefix", "Mistral `[PREFIX]` 11"),
    151660: ("FIM middle", "Mistral `[MIDDLE]` 12"),
    151661: ("FIM suffix", "Mistral `[SUFFIX]` 13"),
    151662: ("FIM padding", "无对应；排除"),
    151663: ("repository name", "无对应；排除"),
    151664: ("file separator", "无对应；排除"),
    151665: ("tool response start", "Mistral `[TOOL_RESULTS]` 7"),
    151666: ("tool response end", "Mistral `[/TOOL_RESULTS]` 8"),
    151667: ("thinking start", "无对应；禁止自动映射"),
    151668: ("thinking end", "无对应；禁止自动映射"),
}


MISTRAL_HANDLING = {
    0: ("UNK", "无 Qwen 控制 token 对应；仅显式 fallback"),
    1: ("BOS", "由 receiver 原生协议注入"),
    2: ("EOS", "Qwen `<|im_end|>` 151645"),
    3: ("user instruction start", "Qwen chat-template 序列；非 one-hot"),
    4: ("user instruction end / response start", "Qwen chat-template 序列；非 one-hot"),
    5: ("available tools start", "Qwen `<tools>` 普通-token 序列"),
    6: ("available tools end", "Qwen `</tools>` 普通-token 序列"),
    7: ("tool results start", "Qwen `<tool_response>` 151665"),
    8: ("tool results end", "Qwen `</tool_response>` 151666"),
    9: ("tool calls start", "Qwen `<tool_call>` 151657（近似）"),
    10: ("reserved PAD", "Qwen `<|endoftext|>` 151643；需显式配置"),
    11: ("FIM prefix", "Qwen `<|fim_prefix|>` 151659"),
    12: ("FIM middle", "Qwen `<|fim_middle|>` 151660"),
    13: ("FIM suffix", "Qwen `<|fim_suffix|>` 151661"),
}


DEEPSEEK_HANDLING = {
    128000: ("BOS", "由 receiver 原生协议注入；Qwen 无 BOS"),
    128001: ("EOS，同时配置为 PAD", "EOS 对应 Qwen `<|im_end|>` 151645；PAD 语义需单独处理"),
    128004: ("fine-tuning right padding", "训练期专用；推理 OT 支撑排除"),
    128006: ("header start", "Qwen chat-template 序列；当前 DeepSeek 模板未使用"),
    128007: ("header end", "Qwen chat-template 序列；当前 DeepSeek 模板未使用"),
    128008: ("end of message", "与 Qwen `<|im_end|>` 功能近似；当前模板未使用"),
    128009: ("end of turn", "与 Qwen `<|im_end|>` 功能近似；当前模板未使用"),
    128010: ("Python tag", "无直接对应；按任务显式处理"),
    128011: ("user role", "Qwen `<|im_start|>user\\n` 模板级序列"),
    128012: ("assistant role", "Qwen `<|im_start|>assistant\\n` 模板级序列"),
    128013: ("thinking start", "Qwen `<think>` 151667；可 one-hot"),
    128014: ("thinking end", "Qwen `</think>` 151668；可 one-hot"),
    128015: ("reserved pad token", "当前未配置为 pad_token；默认排除"),
}


def generate(output: Path) -> None:
    source = AutoTokenizer.from_pretrained(
        SOURCE, revision=SOURCE_REVISION, use_fast=True
    )
    target = AutoTokenizer.from_pretrained(
        TARGET, revision=TARGET_REVISION, use_fast=True
    )
    deepseek = AutoTokenizer.from_pretrained(
        DEEPSEEK, revision=DEEPSEEK_REVISION, use_fast=True
    )
    source_added = added_tokens(source)
    target_added = added_tokens(target)
    deepseek_added = added_tokens(deepseek)

    lines = [
        "# Qwen3-8B 跨模型特殊 Token 对齐说明",
        "",
        "## 1. 范围与版本",
        "",
        "本文档列出 Qwen3-8B、Mistral-Nemo-Instruct-2407 和 DeepSeek-R1-Distill-Llama-8B 固定 tokenizer revision 的全部 added/control token。`backend special` 是 tokenizer backend 的标志；虽未标记 special、但属于原子 added token 的控制标记也纳入清单。",
        "",
        "| 角色 | 模型 | Revision | 总词表 | Added/control token |",
        "|---|---|---|---:|---:|",
        f"| Source A | `{SOURCE}` | `{SOURCE_REVISION}` | {len(source):,} | {len(source_added):,} |",
        f"| Receiver B | `{TARGET}` | `{TARGET_REVISION}` | {len(target):,} | {len(target_added):,} |",
        f"| Alternative B | `{DEEPSEEK}` | `{DEEPSEEK_REVISION}` | {len(deepseek):,} | {len(deepseek_added):,} |",
        "",
        "Mistral 的 ID 14–999 是 986 个预留 `<SPECIAL_n>` 槽。它们没有已注册语义，必须从普通词表对齐、ANN 候选和 OT 有效支撑中排除。",
        "",
        "DeepSeek 的 256 个 added/control token 中有 243 个 `<|reserved_special_token_n|>` 预留槽；这些槽同样必须从普通词表对齐、ANN 候选和 OT 有效支撑中排除。",
        "",
        "### 1.1 Qwen3-8B → DeepSeek tokenizer 对比摘要",
        "",
        "| 指标 | 结果 |",
        "|---|---:|",
        "| Qwen 普通 token 数 | 151,643 |",
        "| DeepSeek 普通 token 数 | 128,000 |",
        "| 共享 exact-byte token | 109,566 |",
        "| Qwen 普通词表 exact-byte 覆盖率 | 72.25% |",
        "| DeepSeek 普通词表 exact-byte 覆盖率 | 85.60% |",
        "| 6 条中英样本的 Qwen token 出现次数 exact 覆盖率 | 90.12% |",
        "| 样本平均 DeepSeek/Qwen token 长度比 | 1.122 |",
        "| 小规模 transport 列 | 61 exact-byte + 8 byte-span |",
        "",
        "英文与代码样本的切分大多一致；中文在 DeepSeek 中通常更碎。数字切分也不同：Qwen 将 `17`、`23` 拆成单个数字，而 DeepSeek 的预切分规则允许 1–3 位数字成组。详细逐样本结果见 `C2C/local/transport/qwen3_8b_to_deepseek_r1_distill_llama_8b_tokenizer_comparison.json`。",
        "",
        "## 2. 功能对应摘要",
        "",
        "| 功能 | Qwen3-8B | Mistral-Nemo | 处理方式 |",
        "|---|---|---|---|",
        "| 响应 EOS | `<\\|im_end\\|>` 151645 | `</s>` 2 | 可 one-hot |",
        "| Padding | `<\\|endoftext\\|>` 151643 | `<pad>` 10 | 条件映射；Mistral 当前未配置 `pad_token` |",
        "| FIM prefix | `<\\|fim_prefix\\|>` 151659 | `[PREFIX]` 11 | 可 one-hot |",
        "| FIM middle | `<\\|fim_middle\\|>` 151660 | `[MIDDLE]` 12 | 可 one-hot |",
        "| FIM suffix | `<\\|fim_suffix\\|>` 151661 | `[SUFFIX]` 13 | 可 one-hot |",
        "| 工具结果开始 | `<tool_response>` 151665 | `[TOOL_RESULTS]` 7 | 可按功能映射 |",
        "| 工具结果结束 | `</tool_response>` 151666 | `[/TOOL_RESULTS]` 8 | 可按功能映射 |",
        "| 工具调用开始 | `<tool_call>` 151657 | `[TOOL_CALLS]` 9 | 近似映射 |",
        "| 用户/助手边界 | `<\\|im_start\\|>` + role + `<\\|im_end\\|>` | `<s>[INST]...[/INST]` | 模板级序列，不做 one-hot |",
        "| Thinking | `<think>`, `</think>` | 无 | 禁止自动映射 |",
        "| 视觉/坐标控制 | 多个 Qwen token | 无 | 排除 |",
        "",
        "### 2.1 Qwen3-8B → DeepSeek 功能对应补充",
        "",
        "| 功能 | Qwen3-8B | DeepSeek-R1-Distill-Llama-8B | 处理方式 |",
        "|---|---|---|---|",
        "| 响应 EOS | `<\\|im_end\\|>` 151645 | `<｜end▁of▁sentence｜>` 128001 | 可按 EOS 功能映射 |",
        "| Thinking start | `<think>` 151667 | `<think>` 128013 | 可 one-hot |",
        "| Thinking end | `</think>` 151668 | `</think>` 128014 | 可 one-hot |",
        "| User role | `<\\|im_start\\|>user\\n` | `<｜User｜>` 128011 | 模板级序列 |",
        "| Assistant role | `<\\|im_start\\|>assistant\\n` | `<｜Assistant｜>` 128012 | 模板级序列 |",
        "| BOS | 无 | `<｜begin▁of▁sentence｜>` 128000 | 由 receiver 注入 |",
        "| 工具边界 | Qwen 原子 tool token | DeepSeek 模板中的普通-token 序列 | 不做 one-hot |",
        "",
        "## 3. Qwen3-8B 完整 added/control token 表",
        "",
        "| ID | Token | Backend special | Tokenizer 配置角色 | 功能 | 建议对应/处理 |",
        "|---:|---|:---:|---|---|---|",
    ]
    for token_id, token in sorted(source_added.items()):
        purpose, handling = QWEN_HANDLING.get(token_id, ("未登记", "人工审计"))
        lines.append(
            f"| {token_id} | {markdown_token(str(token))} | {'yes' if token.special else 'no'} | "
            f"{configured_roles(source, token_id)} | {purpose} | {handling} |"
        )

    lines += [
        "",
        "## 4. Mistral-Nemo-Instruct-2407 完整 special token 表",
        "",
        "| ID | Token | Backend special | Tokenizer 配置角色 | 功能 | 建议对应/处理 |",
        "|---:|---|:---:|---|---|---|",
    ]
    for token_id, token in sorted(target_added.items()):
        if token_id in MISTRAL_HANDLING:
            purpose, handling = MISTRAL_HANDLING[token_id]
        else:
            purpose, handling = "未分配预留槽", "从 ANN/OT 支撑排除，目标边际设为 0"
        lines.append(
            f"| {token_id} | {markdown_token(str(token))} | {'yes' if token.special else 'no'} | "
            f"{configured_roles(target, token_id)} | {purpose} | {handling} |"
        )

    lines += [
        "",
        "## 5. DeepSeek-R1-Distill-Llama-8B 完整 added/control token 表",
        "",
        "| ID | Token | Backend special | Tokenizer 配置角色 | 功能 | 建议对应/处理 |",
        "|---:|---|:---:|---|---|---|",
    ]
    for token_id, token in sorted(deepseek_added.items()):
        if token_id in DEEPSEEK_HANDLING:
            purpose, handling = DEEPSEEK_HANDLING[token_id]
        else:
            purpose, handling = "未分配预留槽", "从 ANN/OT 支撑排除，目标边际设为 0"
        lines.append(
            f"| {token_id} | {markdown_token(str(token))} | {'yes' if token.special else 'no'} | "
            f"{configured_roles(deepseek, token_id)} | {purpose} | {handling} |"
        )

    lines += [
        "",
        "DeepSeek chat template 中的 `<｜tool▁calls▁begin｜>`、`<｜tool▁call▁begin｜>`、`<｜tool▁call▁end｜>`、`<｜tool▁calls▁end｜>`、`<｜tool▁outputs▁begin｜>`、`<｜tool▁output▁begin｜>`、`<｜tool▁output▁end｜>`、`<｜tool▁outputs▁end｜>` 和 `<｜tool▁sep｜>` 并非 tokenizer added token，也不会编码为单一 ID；它们是普通 BPE token 序列，必须按模板级序列处理。",
        "",
        "## 6. STT 实现约束",
        "",
        "1. 普通 message 内容使用 exact-byte、byte-span、ANN 和后续 OT 构建 transport。",
        "2. BOS、chat role 和生成边界由 receiver 的原生协议显式注入，不把 `[INST]`/`[/INST]` 当普通词表列拟合。",
        "3. 可安全对应的 EOS、FIM 和工具结果标记使用注册规则，不交给语料统计覆盖。",
        "4. Qwen 视觉和坐标 token、Mistral 未分配预留槽、DeepSeek 未分配预留槽默认从 OT 有效支撑中排除。Qwen↔DeepSeek 的 `<think>`/`</think>` 使用显式规则。",
        "5. `<pad>` ID 10 虽存在于 Mistral 词表，但当前 tokenizer 没有设置 `pad_token`；启用前必须同时冻结 attention-mask 与 generation 配置。",
        "",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path, default=Path("special_token_alignment.md")
    )
    args = parser.parse_args()
    generate(args.output)
    print(f"Saved special-token documentation to {args.output}")


if __name__ == "__main__":
    main()
