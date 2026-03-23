from __future__ import annotations

from dataclasses import dataclass
import pathlib
from typing import Any, Dict, Optional, cast

import numpy as np
import torch
import yaml

from model.rtc_async.action_expert.runner import ActionExpertRunner, ActionExpertRunnerConfig
from model.rtc_async.pipeline.scheduler import RTCChunkScheduler
from model.rtc_async.qwen3_stream.kv_export import export_selected_kv_cache
from model.rtc_async.qwen3_stream.stream_runner_snapshot import Qwen3VLStreamRunnerSnapshot
from normalization import RTCNormalizer

from .template_qwen3_vla import build_prompt_prefill_text, build_step_assistant_prefix
from .vla_qwen3_rtc import Qwen3RTCVLAEncoder


@dataclass
class RTCOnlineResolvedConfig:
    vision_interval_s: float
    max_context_len: int | None
    selected_layers: list[int]
    inference_delay: int
    execute_horizon: int
    action_expert: Dict[str, Any]


def _load_rtc_async_runtime_config(config_path: str) -> RTCOnlineResolvedConfig:
    """加载在线推理期 RTC 配置并映射到强类型结构。

    参数:
        config_path: `rtc_async_vla.yaml` 路径。

    返回:
        `RTCOnlineResolvedConfig`，用于初始化调度器与动作专家运行参数。
    """
    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    stream = raw.get("stream", {}) or {}
    rtc = raw.get("rtc", {}) or {}
    action_expert = raw.get("action_expert", {}) or {}

    selected_layers = [int(x) for x in stream.get("selected_layers", [])]
    if not selected_layers:
        raise ValueError("rtc_async.stream.selected_layers must be non-empty for online pipeline.")

    return RTCOnlineResolvedConfig(
        vision_interval_s=float(stream.get("vision_interval_s", 0.0)),
        max_context_len=(None if stream.get("max_context_len", None) is None else int(float(stream["max_context_len"]))),
        selected_layers=selected_layers,
        inference_delay=int(rtc.get("inference_delay", 0)),
        execute_horizon=int(rtc.get("execute_horizon", 1)),
        action_expert=action_expert,
    )


class Qwen3RTCVLAOnlinePipeline:
    """
    统一在线入口：
    1) 流式写入 VLM 上下文
    2) 导出 KV/attention 条件
    3) 扩散头采样动作 chunk
    4) RTC 异步调度输出 execute_chunk

    该入口不包含任何在线 token 生成/解码逻辑。
    """

    def __init__(
        self,
        *,
        vla_config_path: Optional[str] = None,
        rtc_config_path: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        """构建在线推理 pipeline，并绑定 VLM 条件与 RTC 调度模块。

        参数:
            vla_config_path: VLA 编码器配置路径；为空时使用默认配置。
            rtc_config_path: RTC 运行配置路径；为空时使用默认配置。
            device: 可选运行设备覆盖值。

        接口对应:
            初始化后可通过 `reset/push_observation/sample_and_schedule`
            形成完整在线闭环。
        """
        if vla_config_path is None:
            vla_config_path = str(pathlib.Path(__file__).resolve().parent.parent / "configs" / "vla_qwen3_rtc.yaml")
        if rtc_config_path is None:
            rtc_config_path = str(pathlib.Path(__file__).resolve().parent.parent / "configs" / "rtc_async_vla.yaml")

        self.encoder = Qwen3RTCVLAEncoder(config_path=vla_config_path)
        if device is not None and device != self.encoder.device:
            runtime_device = str(device)
            self.encoder.device = runtime_device
            model_obj = cast(Any, self.encoder.model)
            model_obj.to(runtime_device)

        self.rtc_cfg = _load_rtc_async_runtime_config(rtc_config_path)

        action_cfg = dict(self.rtc_cfg.action_expert)
        state_dim = int(self.encoder.state_dim)
        action_dim = int(action_cfg["action_dim"])
        horizon = int(action_cfg["horizon"])

        runner_cfg = ActionExpertRunnerConfig(
            state_dim=state_dim,
            action_dim=action_dim,
            horizon=horizon,
            cond_dim=int(self.encoder.kv_cache_dim),
            hidden_size=int(action_cfg.get("hidden_size", 512)),
            num_layers=int(action_cfg.get("num_layers", 8)),
            num_heads=int(action_cfg.get("num_heads", 8)),
            mlp_ratio=float(action_cfg.get("mlp_ratio", 4.0)),
            time_embed_dim=int(action_cfg.get("time_embed_dim", 256)),
            norm_eps=float(action_cfg.get("norm_eps", 1e-6)),
            ffn_multiple_of=int(action_cfg.get("ffn_multiple_of", 256)),
            ffn_dim_multiplier=action_cfg.get("ffn_dim_multiplier", None),
            num_inference_steps=int(action_cfg.get("num_inference_steps", 5)),
        )
        self.action_expert = ActionExpertRunner(runner_cfg).to(self.encoder.device)
        self.action_expert.eval()

        self.scheduler = RTCChunkScheduler(
            horizon=runner_cfg.horizon,
            action_dim=runner_cfg.action_dim,
            device=torch.device(self.encoder.device),
        )
        self._next_step_id = 0

        self.inference_delay = int(self.rtc_cfg.inference_delay)
        self.execute_horizon = int(self.rtc_cfg.execute_horizon)
        self.normalizer: RTCNormalizer | None = None

        tokenizer = cast(Any, self.encoder.processor).tokenizer
        self.runner = Qwen3VLStreamRunnerSnapshot(
            model=self.encoder.model,
            vision_interval_s=self.rtc_cfg.vision_interval_s,
            max_context_len=self.rtc_cfg.max_context_len,
            use_step_eviction=True,
            tokenizer=tokenizer,
        )
        self._latest_state: Optional[torch.Tensor] = None

    @property
    def device(self) -> str:
        """返回 pipeline 当前运行设备，供外部状态构造与 checkpoint 加载使用。"""
        return self.encoder.device

    def set_runtime_schedule_params(self, *, inference_delay: int, execute_horizon: int) -> None:
        """覆盖运行期 RTC 调度参数。

        参数:
            inference_delay: 推理延迟步数 `d`。
            execute_horizon: 每次控制执行长度 `h`。
        """

        self.inference_delay = int(inference_delay)
        self.execute_horizon = int(execute_horizon)

    def load_action_expert_checkpoint(self, checkpoint_path: str, strict: bool = True) -> None:
        """加载动作专家权重到在线 pipeline。

        参数:
            checkpoint_path: checkpoint 文件路径。
            strict: 是否严格匹配参数名。
        """
        payload = torch.load(checkpoint_path, map_location=self.device)
        state_dict = payload.get("action_expert", payload)
        self.action_expert.load_state_dict(state_dict, strict=strict)
        normalization_payload = payload.get("normalization", None)
        if normalization_payload is not None:
            self.normalizer = RTCNormalizer.from_payload(normalization_payload)
        self.action_expert.eval()

    def reset(self, prompt: Optional[str] = None) -> None:
        """重置流式上下文与调度状态，并进行 prompt 预填充。

        参数:
            prompt: 可选任务文本；为空时以 tokenizer EOS 做最小预填充。
        """
        self.runner.reset()
        self.scheduler.reset(batch_size=1)
        self._next_step_id = 0
        self._latest_state = None

        input_ids: torch.LongTensor
        if prompt is None:
            tokenizer = cast(Any, self.encoder.processor).tokenizer
            eos_id = getattr(tokenizer, "eos_token_id", None)
            if eos_id is None:
                raise ValueError("tokenizer.eos_token_id is required for prefill.")
            input_ids = cast(torch.LongTensor, torch.tensor([[eos_id]], device=self.device, dtype=torch.long))
        else:
            prefill_text = build_prompt_prefill_text(str(prompt))
            tokenizer = cast(Any, self.encoder.processor).tokenizer
            encoded = tokenizer(prefill_text, add_special_tokens=False, return_tensors="pt")
            input_ids = cast(torch.LongTensor, encoded["input_ids"].to(self.device))
        self.runner.prefill_text(input_ids=input_ids)

    def push_observation(
        self,
        *,
        frames: np.ndarray | torch.Tensor,
        aux_frames: Optional[np.ndarray | torch.Tensor] = None,
        state: torch.Tensor,
        ts_ms: Optional[int] = None,
        num_frames: int = 4,
    ) -> bool:
        """插入一条观测 step 到流式上下文。

        参数:
            frames: 窗口帧序列。
            state: 当前状态向量，支持 `[Ds]` 或 `[1, Ds]`。
            ts_ms: 可选毫秒时间戳。
            num_frames: 输入窗口帧数。

        返回:
            是否成功插入（受时间门控与上下文策略影响）。
        """
        video = self.encoder._make_video_tensor(frames, num_frames)
        aux_video = self.encoder._make_video_tensor(aux_frames, 1) if aux_frames is not None else None
        latest_state = state.to(self.device)
        if latest_state.dim() == 1:
            latest_state = latest_state.unsqueeze(0)
        self._latest_state = latest_state
        return self.runner.insert_step(
            processor=self.encoder.processor,
            video=video,
            aux_video=aux_video,
            ts=str(ts_ms) if ts_ms is not None else None,
        )

    @torch.inference_mode()
    def sample_and_schedule(
        self,
        *,
        inference_delay: Optional[int] = None,
        execute_horizon: Optional[int] = None,
        kv_cache_key: Optional[tuple[Any, ...]] = None,
        generator: torch.Generator | None = None,
    ) -> Dict[str, torch.Tensor | int]:
        """在当前上下文下采样动作 chunk 并执行 RTC 异步调度。

        参数:
            inference_delay: 可选覆盖默认推理延迟。
            execute_horizon: 可选覆盖默认执行视野。
            kv_cache_key: 可选 KV 缓存键，用于采样缓存复用。
            generator: 随机数生成器。

        返回:
            包含 `action_chunk/execute_chunk/step_id` 等字段的调度结果字典，
            供控制环直接执行 `execute_chunk`。
        """
        if self._latest_state is None:
            raise RuntimeError("No state available. Call push_observation first.")

        delay = int(self.inference_delay if inference_delay is None else inference_delay)
        horizon = int(self.execute_horizon if execute_horizon is None else execute_horizon)

        kv_cache = export_selected_kv_cache(
            past_key_values=self.runner.state.past_key_values,
            selected_layers=self.rtc_cfg.selected_layers,
            clone=False,
        )
        attention_mask = self.runner.state.attention_mask
        prompt_mask = None
        step_mask = None
        if attention_mask is not None:
            prompt_mask = torch.zeros_like(attention_mask, dtype=torch.bool)
            step_mask = torch.zeros_like(attention_mask, dtype=torch.bool)
            prompt_mask[:, : self.runner.prefill_len] = True
            if self.runner.step_spans:
                step_start, step_end = self.runner.step_spans[-1]
                tokenizer = cast(Any, self.encoder.processor).tokenizer
                assistant_len = int(
                    len(
                        tokenizer(
                            build_step_assistant_prefix(),
                            add_special_tokens=False,
                            return_attention_mask=False,
                            return_token_type_ids=False,
                        )["input_ids"]
                    )
                )
                step_end = max(step_start, step_end - assistant_len)
                step_mask[:, step_start:step_end] = True

        sample_state = self._latest_state
        if self.normalizer is not None:
            sample_state = self.normalizer.normalize_state(sample_state)

        sampled_chunk = self.action_expert.sample(
            state=sample_state,
            kv_cache=kv_cache,
            attention_mask=attention_mask,
            prompt_mask=prompt_mask,
            step_mask=step_mask,
            kv_cache_key=kv_cache_key,
            generator=generator,
        )
        if self.normalizer is not None:
            sampled_chunk = self.normalizer.unnormalize_action(sampled_chunk)

        step_id = self._next_step_id
        self._next_step_id += 1
        execute_chunk, _ = self.scheduler.schedule(
            next_chunk=sampled_chunk,
            inference_delay=delay,
            execute_horizon=horizon,
        )

        return {
            "step_id": step_id,
            "inference_delay": delay,
            "execute_horizon": horizon,
            "action_chunk": sampled_chunk,
            "execute_chunk": execute_chunk,
        }
