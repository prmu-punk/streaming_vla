from __future__ import annotations


IM_START = "<|im_start|>"
IM_END = "<|im_end|>"


def build_prompt_prefill_text(prompt: str) -> str:
    """构造会话预填充文本，作为流式上下文的起始 user 段。

    参数:
        prompt: 任务语言指令文本。

    返回:
        满足 Qwen3 聊天模板协议的 user 消息片段，可直接传入 tokenizer。
    """
    return f"{IM_START}user\n{prompt}{IM_END}\n"


def build_step_user_prefix(*, ts_ms: int | None, video_token: str, close_previous_assistant: bool) -> str:
    """构造单个 step 的 user 前缀，匹配“视频+状态”输入接口。

    参数:
        ts_ms: 当前 step 的毫秒时间戳；为空时省略 `<ts>` 标签。
        video_token: 由 processor 提供的视频占位 token。
        close_previous_assistant: 是否先闭合上一个 assistant 段。

    返回:
        以 `<state>` 开口结束的前缀字符串，后续可拼接状态占位内容。
    """
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
    """构造单个 step 的 assistant 前缀，衔接动作输出接口。

    返回:
        以 `<act_bos>` 结束的 assistant 前缀，用于后续动作序列生成或对齐。
    """
    return f"</state>{IM_END}\n{IM_START}assistant\n<act_bos>"
