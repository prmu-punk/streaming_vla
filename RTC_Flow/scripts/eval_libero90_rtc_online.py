from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections import deque
from typing import Any, Dict, List, Optional

import imageio
import numpy as np
import torch
from omegaconf import OmegaConf

ROOT_DIR = str(pathlib.Path(__file__).resolve().parent.parent)
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)


def resolve_project_path(path_str: str) -> str:
    """将相对项目路径解析为绝对路径。

    参数:
        path_str: 配置文件中的路径字符串（绝对或相对）。

    返回:
        可直接用于文件读取的绝对路径。
    """
    p = pathlib.Path(path_str)
    if p.is_absolute():
        return str(p)
    return str(pathlib.Path(ROOT_DIR) / p)

from dataset.libero90_async_dataset import LiberoEpisodeDataset
from model import Qwen3RTCVLAOnlinePipeline
from scripts.rtc_rollout_utils import (
    LiberoEnv,
    _build_state_tensor,
    _initial_frame_window,
    _load_task_init_states,
    _set_init_state,
    _window_array,
    benchmark,
    task_name_to_suite_and_ids,
)


def _find_matching_episode_indices(dataset: LiberoEpisodeDataset, prompt: str) -> List[int]:
    """按语言指令在数据集中检索匹配 episode 索引。

    参数:
        dataset: episode 级数据集实例。
        prompt: 目标任务语言指令。

    返回:
        与 `prompt` 完全匹配的 episode 索引列表。
    """
    out: List[int] = []
    for ep_idx in range(len(dataset)):
        if str(dataset[ep_idx]["prompt"]) == prompt:
            out.append(int(ep_idx))
    return out


def _save_video(path: str, frames: List[np.ndarray], fps: int) -> None:
    """将 rollout 帧序列保存为视频文件。

    参数:
        path: 输出视频路径。
        frames: RGB 帧列表。
        fps: 输出帧率。
    """
    out_path = pathlib.Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out_path, frames, fps=fps)


def run_rtc_online_rollout(
    *,
    pipeline: Qwen3RTCVLAOnlinePipeline,
    cfg,
    task_name: str,
    match_rank: int,
    num_frames: int,
    source_dt_ms: int,
    max_control_cycles: int,
    save_video_path: Optional[str],
) -> Dict[str, Any]:
    """执行一次 RTC 在线闭环 rollout，并返回评估指标。

    接口对应:
    - 输入接口: `pipeline.push_observation` 接收窗口帧+状态，`pipeline.sample_and_schedule`
      输出 `execute_chunk`。
    - 输出接口: 返回任务成功率相关统计和调度参数回显，供 CLI/日志记录。

    参数:
        pipeline: 在线推理 pipeline。
        cfg: OmegaConf 配置对象，包含 dataset/training 字段。
        task_name: LIBERO 任务名。
        match_rank: 同 prompt 匹配样本中的初始状态序号。
        num_frames: 每次推理输入的帧窗口长度。
        source_dt_ms: 控制循环时间步毫秒数。
        max_control_cycles: 最大控制循环轮次。
        save_video_path: 非空时保存可视化视频路径。

    返回:
        指标字典，含 success、步数、实际调度超参数等字段。
    """
    suite_name, task_id, _ = task_name_to_suite_and_ids[task_name]
    task_suite = benchmark.get_benchmark_dict()[suite_name]()
    task = task_suite.get_task(task_id)
    init_states = _load_task_init_states(task)

    dataset = LiberoEpisodeDataset(
        zarr_path=resolve_project_path(str(cfg.dataset.zarr_path)),
        image_key=str(cfg.dataset.image_key),
        action_key=str(cfg.dataset.action_key),
        state_keys=[str(k) for k in cfg.dataset.state_keys],
        prompt_key=str(cfg.dataset.prompt_key),
        max_episodes=cfg.dataset.max_episodes,
    )
    matched = _find_matching_episode_indices(dataset, str(task.language))
    if not matched:
        raise ValueError(f"No dataset episodes matched task prompt: {task.language}")
    if match_rank < 0 or match_rank >= len(matched):
        raise ValueError(f"match_rank out of range: {match_rank}, total_matches={len(matched)}")

    init_idx = min(match_rank, int(init_states.shape[0]) - 1)
    image_key = str(cfg.dataset.image_key)
    state_keys = [str(k) for k in cfg.dataset.state_keys]

    env = LiberoEnv(
        task_name=task_name,
        image_size=128,
        seed=int(cfg.training.seed),
        camera_names=[image_key.replace("_rgb", "")],
        state_ports=state_keys,
        max_episode_steps=550,
    )

    frames_to_save: List[np.ndarray] = []
    try:
        with torch.inference_mode():
            obs, _ = env.reset()
            del obs
            obs = _set_init_state(env, init_states[init_idx])
            prompt = str(obs["prompt"])

            pipeline.reset(prompt=prompt)
            frame_window: deque[np.ndarray] = _initial_frame_window(obs[image_key], num_frames=num_frames)
            if save_video_path is not None:
                frames_to_save.append(env.render())

            success = False
            done = bool(env.done)
            total_env_steps = 0

            for cycle in range(max_control_cycles):
                state = _build_state_tensor(obs, state_keys, device=pipeline.device)
                inserted = pipeline.push_observation(
                    frames=_window_array(frame_window),
                    state=state,
                    ts_ms=total_env_steps * source_dt_ms,
                    num_frames=num_frames,
                )
                if not inserted:
                    continue

                out = pipeline.sample_and_schedule()
                execute_chunk = out["execute_chunk"]
                if not isinstance(execute_chunk, torch.Tensor):
                    raise RuntimeError(f"execute_chunk must be torch.Tensor, got {type(execute_chunk)}")
                execute_np = execute_chunk[0].detach().to("cpu").numpy().astype(np.float32)

                for action in execute_np:
                    obs, reward, done, _, _ = env.step(action)
                    total_env_steps += 1
                    frame_window.append(np.asarray(obs[image_key], dtype=np.uint8))
                    if save_video_path is not None:
                        frames_to_save.append(env.render())
                    if reward >= 1.0:
                        success = True
                    if done:
                        break
                if done:
                    break

            if save_video_path is not None:
                _save_video(save_video_path, frames_to_save, fps=max(1, round(1000 / source_dt_ms)))

            return {
                "task_name": task.name,
                "task_prompt": task.language,
                "init_state_idx": int(init_idx),
                "num_frames": int(num_frames),
                "source_dt_ms": int(source_dt_ms),
                "success": bool(success),
                "done": bool(done),
                "total_env_steps": int(total_env_steps),
                "configured_inference_delay": int(pipeline.inference_delay),
                "configured_execute_horizon": int(pipeline.execute_horizon),
            }
    finally:
        env.close()


def main() -> None:
    """评估脚本 CLI 入口，组装 pipeline 并触发在线 rollout。

    接口对应:
    - 输入接口: 命令行参数传入 checkpoint、任务名与调度配置。
    - 输出接口: 打印 JSON 指标，支持可选视频落盘。
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default="RTC_Flow/configs/train_libero90_async.yaml")
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--match-rank", type=int, default=0)
    parser.add_argument("--num-frames", type=int, default=6)
    parser.add_argument("--max-control-cycles", type=int, default=120)
    parser.add_argument("--inference-delay", type=int, default=None)
    parser.add_argument("--execute-horizon", type=int, default=None)
    parser.add_argument("--save-video", action="store_true")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    OmegaConf.resolve(cfg)

    pipeline = Qwen3RTCVLAOnlinePipeline(
        vla_config_path=resolve_project_path(str(cfg.model.vla_config_path)),
        rtc_config_path=resolve_project_path(str(cfg.rtc_async.config_path)),
    )
    pipeline.load_action_expert_checkpoint(str(args.checkpoint), strict=False)
    if args.inference_delay is not None and args.execute_horizon is not None:
        pipeline.set_runtime_schedule_params(
            inference_delay=int(args.inference_delay),
            execute_horizon=int(args.execute_horizon),
        )

    video_path = None
    if args.save_video:
        video_path = str(pathlib.Path.cwd() / "videos" / "rtc_online" / str(args.task) / f"match_{int(args.match_rank):03d}.mp4")

    metrics = run_rtc_online_rollout(
        pipeline=pipeline,
        cfg=cfg,
        task_name=str(args.task),
        match_rank=int(args.match_rank),
        num_frames=int(args.num_frames),
        source_dt_ms=int(cfg.training.source_dt_ms),
        max_control_cycles=int(args.max_control_cycles),
        save_video_path=video_path,
    )
    if video_path is not None:
        metrics["video_path"] = video_path
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
