from __future__ import annotations


IM_START = "<|im_start|>"
IM_END = "<|im_end|>"


def build_prompt_prefill_text(prompt: str) -> str:
    return f"{IM_START}user\n{prompt}{IM_END}\n"


def build_step_user_prefix(*, ts_ms: int | None, video_token: str, close_previous_assistant: bool) -> str:
    parts: list[str] = []
    if close_previous_assistant:
        parts.append(f"{IM_END}\n")
    parts.append(f"{IM_START}user\n")
    parts.append("<step>")
    if ts_ms is not None:
        parts.append(f"<ts>{int(ts_ms)}</ts>")
    parts.append(video_token)
    parts.append("<state>")
    return "".join(parts)


def build_step_assistant_prefix() -> str:
    return f"</state>{IM_END}\n{IM_START}assistant\n<act_bos>"
