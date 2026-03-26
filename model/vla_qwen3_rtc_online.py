from __future__ import annotations

from dataclasses import dataclass
import pathlib
import time
from typing import Any, Dict, Optional, cast

import numpy as np
import torch
import yaml

from model.qwen3_vl.stream_runner import Qwen3VLStreamRunner
from model.rtc_async.action_expert.runner import ActionExpertRunner, ActionExpertRunnerConfig
from model.rtc_async.pipeline import (
    RTCDiTStage,
    RTCExecutionStage,
    RTCThreadedPipelineRunner,
    RTCPipelineQueues,
    RTCVLMStage,
    StepPacket,
)
from model.rtc_async.pipeline.scheduler import RTCChunkScheduler
from normalization import RTCNormalizer

from .template_qwen3_vla import build_prompt_prefill_text
from .vla_qwen3_rtc import Qwen3RTCVLAEncoder


@dataclass
class RTCOnlineResolvedConfig:
    vision_interval_s: float
    max_context_len: int | None
    selected_layers: list[int]
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
    action_expert = raw.get("action_expert", {}) or {}

    selected_layers = [int(x) for x in stream.get("selected_layers", [])]
    if not selected_layers:
        raise ValueError("rtc_async.stream.selected_layers must be non-empty for online pipeline.")

    return RTCOnlineResolvedConfig(
        vision_interval_s=float(stream.get("vision_interval_s", 0.0)),
        max_context_len=(None if stream.get("max_context_len", None) is None else int(float(stream["max_context_len"]))),
        selected_layers=selected_layers,
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
        vlm_device: Optional[str] = None,
        dit_device: Optional[str] = None,
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
        resolved_vlm_device = vlm_device if vlm_device is not None else device
        if resolved_vlm_device is not None and resolved_vlm_device != self.encoder.device:
            runtime_device = str(resolved_vlm_device)
            self.encoder.device = runtime_device
            model_obj = cast(Any, self.encoder.model)
            model_obj.to(runtime_device)
        self.vlm_device = str(self.encoder.device)

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
        self.dit_device = str(dit_device if dit_device is not None else self.vlm_device)
        if self.dit_device != self.vlm_device:
            self.action_expert.to(self.dit_device)
        self.action_expert.eval()

        self.scheduler = RTCChunkScheduler(
            horizon=runner_cfg.horizon,
            action_dim=runner_cfg.action_dim,
            device=torch.device(self.dit_device),
        )
        self._next_step_id = 0
        self.source_dt_ms = 50
        self._last_step_ts_ms: int | None = None
        self.normalizer: RTCNormalizer | None = None

        tokenizer = cast(Any, self.encoder.processor).tokenizer
        self.runner = Qwen3VLStreamRunner(
            model=self.encoder.model,
            vision_interval_s=self.rtc_cfg.vision_interval_s,
            max_context_len=self.rtc_cfg.max_context_len,
            use_step_eviction=True,
            tokenizer=tokenizer,
        )
        self.vlm_stage = RTCVLMStage(
            encoder=self.encoder,
            runner=self.runner,
            processor=self.encoder.processor,
            selected_layers=self.rtc_cfg.selected_layers,
        )
        self.dit_stage = RTCDiTStage(action_expert=self.action_expert)
        self.execution_stage = RTCExecutionStage(scheduler=self.scheduler)
        self.queues = RTCPipelineQueues()
        self.threaded_runner: RTCThreadedPipelineRunner | None = None

    @property
    def device(self) -> str:
        """返回 pipeline 当前运行设备，供外部状态构造与 checkpoint 加载使用。"""
        return self.vlm_device

    def set_runtime_timebase(self, *, source_dt_ms: int) -> None:
        """设置在线 step 时间基准，用于把 `ts_ms` 换算为 `step_delay_steps`。"""
        if int(source_dt_ms) <= 0:
            raise ValueError(f"source_dt_ms must be positive, got {source_dt_ms}")
        self.source_dt_ms = int(source_dt_ms)

    def start_async_pipeline(self, *, poll_interval_s: float = 0.001) -> None:
        """启动单卡线程化 stage pipeline。"""
        if self.threaded_runner is not None and self.threaded_runner.running():
            raise RuntimeError("async pipeline is already running")
        self.threaded_runner = RTCThreadedPipelineRunner(
            queues=self.queues,
            step_to_context=self._drain_step_to_context,
            context_to_action=self._drain_context_to_action,
            action_to_execute=self._drain_action_to_execute,
            poll_interval_s=float(poll_interval_s),
        )
        self.threaded_runner.start()

    def stop_async_pipeline(self) -> None:
        """停止单卡线程化 stage pipeline。"""
        if self.threaded_runner is None:
            return
        self.threaded_runner.stop()
        self.threaded_runner = None

    def async_pipeline_running(self) -> bool:
        return self.threaded_runner is not None and self.threaded_runner.running()

    def load_action_expert_checkpoint(self, checkpoint_path: str, strict: bool = True) -> None:
        """加载动作专家权重到在线 pipeline。

        参数:
            checkpoint_path: checkpoint 文件路径。
            strict: 是否严格匹配参数名。
        """
        payload = torch.load(checkpoint_path, map_location="cpu")
        vla_state_dict = payload.get("vla", None)
        if vla_state_dict is not None:
            self.encoder.load_state_dict(vla_state_dict, strict=False)
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
        if self.async_pipeline_running():
            raise RuntimeError("reset is not allowed while async pipeline is running; stop it first.")
        self.runner.reset()
        self.scheduler.reset(batch_size=1)
        self._next_step_id = 0
        self._last_step_ts_ms = None
        self.queues.clear()

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
        now = time.monotonic()
        if now - self.runner.state.last_vision_time < self.rtc_cfg.vision_interval_s:
            return False
        latest_state = state.to(self.device)
        if latest_state.dim() == 1:
            latest_state = latest_state.unsqueeze(0)
        self.queues.step_queue.put_latest(
            StepPacket(
            step_id=self._next_step_id,
            frames=frames,
            aux_frames=aux_frames,
            state=latest_state,
            ts_ms=ts_ms,
            num_frames=num_frames,
            )
        )
        self._next_step_id += 1
        return True

    def _drain_step_to_context(self) -> Any:
        step_packet = self.queues.step_queue.pop()
        if step_packet is None:
            return None
        packet = self.vlm_stage(step_packet)
        self.queues.context_queue.put_latest(packet)
        return packet

    def _drain_context_to_action(
        self,
        *,
        kv_cache_key: Optional[tuple[Any, ...]] = None,
        generator: torch.Generator | None = None,
    ) -> Any:
        context_packet = self.queues.context_queue.pop()
        if context_packet is None:
            return None
        packet = self.dit_stage(
            context_packet,
            normalizer=self.normalizer,
            kv_cache_key=kv_cache_key,
            generator=generator,
        )
        self.queues.action_queue.put_latest(packet)
        return packet

    def _resolve_step_delay_steps(self, ts_ms: int | None) -> int:
        if ts_ms is None:
            return int(self.scheduler.horizon)
        if self._last_step_ts_ms is None:
            self._last_step_ts_ms = int(ts_ms)
            return int(self.scheduler.horizon)
        delta_ms = max(int(ts_ms) - int(self._last_step_ts_ms), 0)
        self._last_step_ts_ms = int(ts_ms)
        delay_steps = max(1, int(round(float(delta_ms) / float(self.source_dt_ms))))
        return int(delay_steps)

    def _drain_action_to_execute(self) -> Any:
        action_packet = self.queues.action_queue.pop()
        if action_packet is None:
            return None
        step_delay_steps = self._resolve_step_delay_steps(action_packet.ts_ms)
        packet = self.execution_stage(
            action_packet,
            step_delay_steps=step_delay_steps,
        )
        self.queues.execute_queue.put_latest(packet)
        return packet

    def poll_execute_packet(self) -> Optional[Dict[str, torch.Tensor | int]]:
        execute_packet = self.queues.execute_queue.pop()
        if execute_packet is None:
            return None
        return {
            "step_id": execute_packet.step_id,
            "step_delay_steps": execute_packet.step_delay_steps,
            "prefix_len": execute_packet.prefix_len,
            "action_chunk": execute_packet.action_chunk,
            "stitched_chunk": execute_packet.stitched_chunk,
            "execute_chunk": execute_packet.execute_chunk,
        }

    @torch.inference_mode()
    def sample_and_schedule(
        self,
        *,
        kv_cache_key: Optional[tuple[Any, ...]] = None,
        generator: torch.Generator | None = None,
    ) -> Dict[str, torch.Tensor | int]:

        if self.queues.step_queue.empty():
            raise RuntimeError("No pending observation available. Call push_observation first.")
        if self.async_pipeline_running():
            raise RuntimeError(
                "sample_and_schedule is not supported while async pipeline is running; use poll_execute_packet."
            )

        self._drain_step_to_context()
        self._drain_context_to_action(
            kv_cache_key=kv_cache_key,
            generator=generator,
        )
        self._drain_action_to_execute()
        execute_packet = self.queues.execute_queue.pop()
        if execute_packet is None:
            raise RuntimeError("RTC pipeline produced no execute packet.")

        return {
            "step_id": execute_packet.step_id,
            "step_delay_steps": execute_packet.step_delay_steps,
            "prefix_len": execute_packet.prefix_len,
            "action_chunk": execute_packet.action_chunk,
            "stitched_chunk": execute_packet.stitched_chunk,
            "execute_chunk": execute_packet.execute_chunk,
        }
