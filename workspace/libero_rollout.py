from __future__ import annotations

from collections import deque
import os
import pathlib
import sys
import time
from typing import Any, Dict, List, Optional

import imageio
import numpy as np
import torch
import yaml
from omegaconf import DictConfig
from tqdm.auto import tqdm


ROOT_DIR = str(pathlib.Path(__file__).resolve().parent.parent)
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)
OAT_ROOT = str(pathlib.Path(ROOT_DIR) / "oat")
if OAT_ROOT not in sys.path:
    sys.path.append(OAT_ROOT)
LIBERO_ROOT = str(pathlib.Path(ROOT_DIR) / "oat" / "third_party" / "LIBERO")
if LIBERO_ROOT not in sys.path:
    sys.path.append(LIBERO_ROOT)


def _ensure_libero_config() -> None:
    libero_config_root = pathlib.Path(ROOT_DIR) / ".libero"
    libero_config_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("LIBERO_CONFIG_PATH", str(libero_config_root))

    config_path = libero_config_root / "config.yaml"
    if config_path.exists():
        return

    benchmark_root = pathlib.Path(LIBERO_ROOT) / "libero" / "libero"
    config = {
        "benchmark_root": str(benchmark_root),
        "bddl_files": str(benchmark_root / "bddl_files"),
        "init_states": str(benchmark_root / "init_files"),
        "datasets": str(pathlib.Path(LIBERO_ROOT) / "libero" / "datasets"),
        "assets": str(benchmark_root / "assets"),
    }
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f)


_ensure_libero_config()


from libero.libero import benchmark
from libero.libero import get_libero_path
from model.vla_qwen3 import Qwen3VLA
from oat.oat.env.libero.env import LiberoEnv, task_name_to_suite_and_ids
from utils.vla_utils import module_device


def infer_chunk_horizon(vla: Qwen3VLA, fixed_action_tokens: int) -> int:
    allowed = vla.action_tokenizer.allowed_hf_token_ids(device=module_device(vla.model), include_eos=False)
    if allowed.numel() == 0:
        raise ValueError("No allowed action token ids found.")
    probe = allowed[0].view(1, 1).repeat(1, fixed_action_tokens)
    with torch.no_grad():
        chunk = vla.action_tokenizer.detokenize(probe)
    if chunk.dim() != 3:
        raise ValueError(f"detokenized chunk must be [B, H, D], got {tuple(chunk.shape)}")
    return int(chunk.shape[1])


def _sync_device(device: str) -> None:
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def _timed_call(device: str, fn, *args, **kwargs):
    _sync_device(device)
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    _sync_device(device)
    dt = time.perf_counter() - t0
    return out, dt


def _build_state_tensor(obs: Dict[str, Any], state_keys: List[str], device: str) -> torch.Tensor:
    pieces = []
    for key in state_keys:
        arr = np.asarray(obs[key], dtype=np.float32).reshape(-1)
        pieces.append(arr)
    state = np.concatenate(pieces, axis=0)
    return torch.from_numpy(state).to(device=device).unsqueeze(0)


def _initial_frame_window(frame: np.ndarray, num_frames: int) -> deque[np.ndarray]:
    q: deque[np.ndarray] = deque(maxlen=num_frames)
    for _ in range(num_frames):
        q.append(np.asarray(frame, dtype=np.uint8))
    return q


def _window_array(frame_window: deque[np.ndarray]) -> np.ndarray:
    return np.stack(list(frame_window), axis=0)


def _set_init_state(env: LiberoEnv, init_state: np.ndarray) -> Dict[str, Any]:
    raw_obs = env.env.set_init_state(init_state)
    env.done = False
    env.cur_step = 0
    return env._extract_obs(raw_obs)


def _warmup_env(env: LiberoEnv, n_steps: int) -> Dict[str, Any]:
    obs = None
    zero = np.zeros(env.action_space.shape, dtype=np.float32)
    for _ in range(n_steps):
        obs, _, done, _, _ = env.step(zero)
        if done:
            break
    if obs is None:
        obs, _ = env.reset()
    return obs


def _load_task_init_states(task) -> torch.Tensor:
    init_states_path = pathlib.Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file
    return torch.load(init_states_path, weights_only=False)


def rollout_one_episode(
    *,
    vla: Qwen3VLA,
    env: LiberoEnv,
    init_state: np.ndarray,
    image_key: str,
    state_keys: List[str],
    num_frames: int,
    fixed_action_tokens: int,
    source_dt_ms: int,
    temperature: float = 0.0,
    top_k: Optional[int] = None,
    warmup_steps: int = 5,
    save_video_path: Optional[str] = None,
) -> Dict[str, Any]:
    device = module_device(vla.model)
    obs, _ = env.reset()
    del obs
    obs = _set_init_state(env, init_state)
    obs = _warmup_env(env, warmup_steps)

    prompt = str(obs["prompt"])
    runner = vla.new_runner()
    vla.prefill(runner, prompt)

    frame_window = _initial_frame_window(obs[image_key], num_frames=num_frames)
    frames_to_save: List[np.ndarray] = []
    if save_video_path is not None:
        frames_to_save.append(env.render())

    step_idx = warmup_steps
    success = False
    executed_actions = 0
    done = bool(env.done)
    insert_step_time = 0.0
    generate_time = 0.0
    env_step_time = 0.0
    decision_cycle_times: List[float] = []
    decision_count = 0

    while not done:
        decision_t0 = time.perf_counter()
        state = _build_state_tensor(obs, state_keys, device=device)
        inserted, insert_dt = _timed_call(
            device,
            vla.insert_step,
            runner,
            _window_array(frame_window),
            state=state,
            ts=step_idx * source_dt_ms,
            num_frames=num_frames,
            source_dt_ms=source_dt_ms,
        )
        insert_step_time += insert_dt
        if not inserted:
            raise RuntimeError("Failed to insert rollout step into runner.")

        gen, generate_dt = _timed_call(
            device,
            vla.generate_action_chunk,
            runner,
            fixed_action_tokens=fixed_action_tokens,
            temperature=temperature,
            top_k=top_k,
        )
        generate_time += generate_dt
        decision_count += 1
        action_chunk = gen["action_chunk"]
        if action_chunk is None:
            raise RuntimeError("Model returned no action chunk.")
        action_chunk_np = action_chunk[0].detach().to("cpu").numpy().astype(np.float32)

        for action in action_chunk_np:
            step_out, step_dt = _timed_call(device, env.step, action)
            obs, reward, done, _, _ = step_out
            env_step_time += step_dt
            executed_actions += 1
            step_idx += 1
            frame_window.append(np.asarray(obs[image_key], dtype=np.uint8))
            if save_video_path is not None:
                frames_to_save.append(env.render())
            if reward >= 1.0:
                success = True
            if done:
                break

        decision_cycle_times.append(time.perf_counter() - decision_t0)

        if done:
            break

    if save_video_path is not None:
        out_path = pathlib.Path(save_video_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(out_path, frames_to_save, fps=max(1, round(1000 / source_dt_ms)))

    return {
        "success": bool(success),
        "executed_actions": int(executed_actions),
        "prompt": prompt,
        "mean_decision_cycle_time_s": float(np.mean(decision_cycle_times)) if decision_cycle_times else 0.0,
    }


def evaluate_libero_rollouts(
    *,
    vla: Qwen3VLA,
    cfg: DictConfig,
    task_name: str,
    n_eval: int,
    save_videos: bool = False,
    output_dir: str | None = None,
    temperature: float = 0.0,
    top_k: Optional[int] = None,
) -> Dict[str, Any]:
    if task_name not in task_name_to_suite_and_ids:
        raise ValueError(f"Unknown LIBERO task: {task_name}")

    suite_name, task_id, _ = task_name_to_suite_and_ids[task_name]
    task_suite = benchmark.get_benchmark_dict()[suite_name]()
    task = task_suite.get_task(task_id)
    init_states = _load_task_init_states(task)
    n_eval = min(int(n_eval), int(init_states.shape[0]))

    image_key = str(cfg.dataset.image_key)
    state_keys = [str(k) for k in cfg.dataset.state_keys]
    num_frames = int(cfg.model.num_frames)
    fixed_action_tokens = int(cfg.model.fixed_action_tokens)
    source_dt_ms = int(cfg.training.source_dt_ms)

    env = LiberoEnv(
        task_name=task_name,
        image_size=128,
        seed=int(cfg.training.seed),
        camera_names=[image_key.replace("_rgb", "")],
        state_ports=state_keys,
        max_episode_steps=550,
    )

    results: List[Dict[str, Any]] = []
    progress = tqdm(range(n_eval), desc=f"Rollout {task_name}", dynamic_ncols=True)
    try:
        for ep_idx in progress:
            video_path = None
            if save_videos:
                base_dir = pathlib.Path(output_dir or pathlib.Path.cwd())
                video_path = str(base_dir / "videos" / task_name / f"episode_{ep_idx:03d}.mp4")
            episode = rollout_one_episode(
                vla=vla,
                env=env,
                init_state=init_states[ep_idx],
                image_key=image_key,
                state_keys=state_keys,
                num_frames=num_frames,
                fixed_action_tokens=fixed_action_tokens,
                source_dt_ms=source_dt_ms,
                temperature=temperature,
                top_k=top_k,
                save_video_path=video_path,
            )
            episode["episode_idx"] = int(ep_idx)
            results.append(episode)
            success_rate = float(np.mean([float(x["success"]) for x in results]))
            progress.set_postfix(success_rate=f"{success_rate:.3f}")
    finally:
        env.close()

    return {
        "task_name": task.name,
        "task_prompt": task.language,
        "n_eval": int(n_eval),
        "success_rate": float(np.mean([float(x["success"]) for x in results])) if results else 0.0,
        "mean_decision_cycle_time_s": (
            float(np.mean([float(x["mean_decision_cycle_time_s"]) for x in results])) if results else 0.0
        ),
        "episodes": results,
    }
