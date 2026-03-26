from __future__ import annotations


IM_START = "<|im_start|>"
IM_END = "<|im_end|>"


def build_prompt_prefill_text(prompt: str) -> str:
    return f"{IM_START}user\n{prompt}{IM_END}\n"


def build_video_text(*, video_token: str, has_aux: bool) -> str:
    if has_aux:
        return f"main view: {video_token} wrist view: {video_token}"
    return video_token


def build_step_user_prefix(*, ts_ms: int | None, video_token: str) -> str:
    parts: list[str] = [f"{IM_START}user\n"]
    if ts_ms is not None:
        parts.append(f"time: {float(ts_ms) / 1000.0:.2f}s\n")
    parts.append(f"obs: {video_token}")
    parts.append(f"{IM_END}\n")
    return "".join(parts)
