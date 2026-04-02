from __future__ import annotations

import argparse
import json
import pathlib
from typing import Any, Optional

import torch
from omegaconf import OmegaConf

from model import Qwen3RTCVLAOnlinePipeline
from model.rtc_async.pipeline.pipeline_types import ExecutePacket

from rollout_libero_rtc_async_video import run_async_rollout


class Qwen3NoRTCAsyncVLAOnlinePipeline(Qwen3RTCVLAOnlinePipeline):
    def _drain_context_to_action(
        self,
        *,
        kv_cache_key: Optional[tuple[Any, ...]] = None,
        generator: torch.Generator | None = None,
    ) -> Any:
        context_packet = self.queues.context_queue.pop()
        if context_packet is None:
            return None
        step_delay_steps = self._resolve_step_delay_steps(context_packet.ts_ms)
        packet = self.dit_stage(
            context_packet,
            step_delay_steps=step_delay_steps,
            known_action=None,
            known_mask=None,
            normalizer=self.normalizer,
            kv_cache_key=kv_cache_key,
            generator=generator,
        )
        self.queues.action_queue.put_latest(packet)
        return packet

    def _drain_action_to_execute(self) -> Any:
        action_packet = self.queues.action_queue.pop()
        if action_packet is None:
            return None
        execute_steps = min(max(int(action_packet.step_delay_steps), 0), self.scheduler.horizon)
        stitched_chunk = action_packet.action_chunk
        execute_chunk = stitched_chunk[:, :execute_steps]
        self.scheduler.last_stitched_chunk = stitched_chunk.to(
            device=self.scheduler.device,
            dtype=self.scheduler.dtype,
        )
        packet = ExecutePacket(
            step_id=action_packet.step_id,
            ts_ms=action_packet.ts_ms,
            step_delay_steps=int(action_packet.step_delay_steps),
            prefix_len=0,
            action_chunk=action_packet.action_chunk,
            stitched_chunk=stitched_chunk,
            execute_chunk=execute_chunk,
        )
        self.queues.execute_queue.put_latest(packet)
        return packet


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a single async non-RTC rollout on LIBERO and save a video."
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/train_libero90_async.yaml")
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--vlm-device", type=str, default=None)
    parser.add_argument("--dit-device", type=str, default=None)
    parser.add_argument("--match-rank", type=int, default=0)
    parser.add_argument("--num-frames", type=int, default=None)
    parser.add_argument("--max-env-steps", type=int, default=900)
    parser.add_argument("--observation-gap-steps", type=int, default=None)
    parser.add_argument("--step-dt-min-ms", type=int, default=None)
    parser.add_argument("--step-dt-max-ms", type=int, default=None)
    parser.add_argument("--max-wait-s", type=float, default=2.0)
    parser.add_argument("--video-path", type=str, default=None)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    OmegaConf.resolve(cfg)

    pipeline = Qwen3NoRTCAsyncVLAOnlinePipeline(
        vla_config_path=str(cfg.model.vla_config_path),
        rtc_config_path=str(cfg.rtc_async.config_path),
        vlm_device=args.vlm_device,
        dit_device=args.dit_device,
    )
    pipeline.load_action_expert_checkpoint(str(args.checkpoint), strict=False)

    num_frames = int(cfg.model.get("num_frames", 1) if args.num_frames is None else args.num_frames)
    video_path = args.video_path
    if video_path is None:
        video_path = str(
            pathlib.Path.cwd() / "videos" / "no_rtc_async_rollout" / str(args.task) / f"match_{int(args.match_rank):03d}.mp4"
        )

    metrics = run_async_rollout(
        pipeline=pipeline,
        cfg=cfg,
        task_name=str(args.task),
        match_rank=int(args.match_rank),
        num_frames=num_frames,
        source_dt_ms=int(cfg.training.source_dt_ms),
        observation_gap_steps=(None if args.observation_gap_steps is None else int(args.observation_gap_steps)),
        step_dt_min_ms=int(cfg.training.step_dt_min_ms if args.step_dt_min_ms is None else args.step_dt_min_ms),
        step_dt_max_ms=int(cfg.training.step_dt_max_ms if args.step_dt_max_ms is None else args.step_dt_max_ms),
        max_env_steps=int(args.max_env_steps),
        save_video_path=str(video_path),
        max_wait_s=float(args.max_wait_s),
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
