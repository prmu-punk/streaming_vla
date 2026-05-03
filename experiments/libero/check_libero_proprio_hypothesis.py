from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from experiments.libero.build_xt_replay import (  # noqa: E402
    _mixed_precision_to_model_dtype,
    _normalize_mixed_precision,
    resolve_weight_checkpoint,
)
from experiments.libero.eval_libero_policy_utils import (  # noqa: E402
    _normalize_proprio,
    _obs_to_model_input,
    _postprocess_libero_action_chunk,
)
from experiments.libero.libero_utils import LIBERO_ENV_RESOLUTION, get_libero_env  # noqa: E402
from fastwam.datasets.lerobot.processors.fastwam_processor import FastWAMProcessor  # noqa: E402
from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json  # noqa: E402
from fastwam.datasets.lerobot.utils.rotation import matrix_to_euler_angles, quaternion_to_matrix  # noqa: E402
from fastwam.utils.pytorch_utils import set_global_seed  # noqa: E402
from libero.libero import benchmark  # noqa: E402


def _register_resolver_if_needed(name: str, fn) -> None:
    if not OmegaConf.has_resolver(name):
        OmegaConf.register_new_resolver(name, fn)


_register_resolver_if_needed("eval", eval)
_register_resolver_if_needed("max", lambda x: max(x))
_register_resolver_if_needed("split", lambda s, idx: s.split("/")[int(idx)])

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare axis-angle vs RPY proprio encoding on one LIBERO obs.")
    parser.add_argument("--task-config", default="libero_streaming_action_ft_2cam224_1e-4")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--task-suite-name", default="libero_10")
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--trial-idx", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--mixed-precision", default="bf16")
    parser.add_argument("--output-json", default=None)
    return parser.parse_args()


def _compose_cfg(args: argparse.Namespace):
    overrides = [
        f"task={args.task_config}",
        f"ckpt={args.ckpt}",
        f"mixed_precision={args.mixed_precision}",
        f"seed={int(args.seed)}",
        f"EVALUATION.device={args.device}",
    ]
    config_dir = str((project_root / "configs").resolve())
    with initialize_config_dir(version_base="1.3", config_dir=config_dir):
        return compose(config_name="sim_libero", overrides=overrides)


def _build_model(cfg, *, ckpt: str, model_dtype: torch.dtype, device: str):
    model = instantiate(cfg.model, model_dtype=model_dtype, device=device)
    weight_ckpt = resolve_weight_checkpoint(ckpt)
    model.load_checkpoint(str(weight_ckpt))
    return model.to(device).eval()


def _resolve_dataset_stats_path(cfg) -> Path:
    explicit = cfg.EVALUATION.get("dataset_stats_path")
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(Path(os.path.expanduser(os.path.expandvars(str(explicit)))))
    if cfg.get("ckpt") is not None:
        ckpt = Path(os.path.expanduser(os.path.expandvars(str(cfg.ckpt))))
        for parent in list(ckpt.parents)[:4]:
            candidates.append(parent / "dataset_stats.json")
    candidates.append(project_root / "checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json")
    for path in candidates:
        if path.exists():
            return path.resolve()
    raise FileNotFoundError("Failed to locate dataset_stats.json")


def _quat_xyzw_to_euler_xyz(quat_xyzw: np.ndarray) -> np.ndarray:
    quat_xyzw = np.asarray(quat_xyzw, dtype=np.float32)
    quat_wxyz = torch.tensor(
        [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]],
        dtype=torch.float32,
    )
    euler = matrix_to_euler_angles(quaternion_to_matrix(quat_wxyz), convention="XYZ")
    return euler.detach().cpu().numpy().astype(np.float32)


def _extract_sim_state_rpy(obs: dict[str, Any]) -> np.ndarray:
    return np.concatenate(
        (
            np.asarray(obs["robot0_eef_pos"], dtype=np.float32),
            _quat_xyzw_to_euler_xyz(obs["robot0_eef_quat"]),
            np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32),
        )
    ).astype(np.float32)


@torch.no_grad()
def main() -> None:
    args = _parse_args()
    set_global_seed(max(int(args.seed), 1), get_worker_init_fn=False)
    cfg = _compose_cfg(args)

    precision = _normalize_mixed_precision(str(cfg.get("mixed_precision", args.mixed_precision)))
    model_dtype = _mixed_precision_to_model_dtype(precision)
    device = str(args.device)
    model = _build_model(cfg, ckpt=str(args.ckpt), model_dtype=model_dtype, device=device)

    dataset_stats_path = _resolve_dataset_stats_path(cfg)
    dataset_stats = load_dataset_stats_from_json(str(dataset_stats_path))
    processor: FastWAMProcessor = instantiate(cfg.data.train.processor).eval()
    processor.set_normalizer_from_stats(dataset_stats)

    video_size = cfg.data.train.get("video_size", [224, 224])
    input_h = int(video_size[0])
    input_w = int(video_size[1])
    action_horizon_cfg = cfg.EVALUATION.get("action_horizon", None)
    action_horizon = int(cfg.data.train.num_frames) - 1 if action_horizon_cfg is None else int(action_horizon_cfg)
    num_inference_steps = int(cfg.EVALUATION.get("num_inference_steps", cfg.get("eval_num_inference_steps", 10)))
    sigma_shift = None if cfg.EVALUATION.get("sigma_shift") is None else float(cfg.EVALUATION.get("sigma_shift"))

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[str(args.task_suite_name)]()
    task = task_suite.get_task(int(args.task_id))
    initial_states = list(task_suite.get_task_init_states(int(args.task_id)))
    while len(initial_states) <= int(args.trial_idx):
        initial_states.extend(initial_states[: (int(args.trial_idx) + 1 - len(initial_states))])

    env, task_description = get_libero_env(
        task,
        LIBERO_ENV_RESOLUTION,
        cfg.get("seed"),
        render_gpu_device_id=int(cfg.get("gpu_id", 0)),
    )
    try:
        env.reset()
        obs = env.set_init_state(initial_states[int(args.trial_idx)])

        image, proprio_axis, _imgs = _obs_to_model_input(
            obs,
            cfg=cfg,
            processor=processor,
            width=input_w,
            height=input_h,
            device="cpu",
            dtype=torch.float32,
        )
        proprio_rpy = _normalize_proprio(_extract_sim_state_rpy(obs), processor)

        prompt = (
            "A video recorded from a robot's point of view executing the following instruction: "
            f"{task_description}"
        )
        out_axis = model.infer_action_streaming(
            prompt=prompt,
            input_image=image,
            action_horizon=action_horizon,
            proprio=proprio_axis,
            num_inference_steps=num_inference_steps,
            sigma_shift=sigma_shift,
            seed=int(args.seed),
            rand_device=device,
            tiled=bool(cfg.EVALUATION.get("tiled", False)),
        )
        model.reset_streaming_state()
        out_rpy = model.infer_action_streaming(
            prompt=prompt,
            input_image=image,
            action_horizon=action_horizon,
            proprio=proprio_rpy,
            num_inference_steps=num_inference_steps,
            sigma_shift=sigma_shift,
            seed=int(args.seed),
            rand_device=device,
            tiled=bool(cfg.EVALUATION.get("tiled", False)),
        )

        action_axis = _postprocess_libero_action_chunk(out_axis["action"], processor=processor, cfg=cfg)
        action_rpy = _postprocess_libero_action_chunk(out_rpy["action"], processor=processor, cfg=cfg)

        diff = np.asarray(action_rpy, dtype=np.float32) - np.asarray(action_axis, dtype=np.float32)
        result = {
            "task_suite_name": str(args.task_suite_name),
            "task_id": int(args.task_id),
            "trial_idx": int(args.trial_idx),
            "task_description": str(task_description),
            "num_inference_steps": int(num_inference_steps),
            "action_horizon": int(action_horizon),
            "proprio_axis": np.asarray(proprio_axis, dtype=np.float32).tolist(),
            "proprio_rpy": np.asarray(proprio_rpy, dtype=np.float32).tolist(),
            "action_axis_first": np.asarray(action_axis[0], dtype=np.float32).tolist(),
            "action_rpy_first": np.asarray(action_rpy[0], dtype=np.float32).tolist(),
            "chunk_mse": float(np.mean(diff ** 2)),
            "chunk_mae": float(np.mean(np.abs(diff))),
            "per_dim_mse": np.mean(diff ** 2, axis=0).astype(float).tolist(),
            "gripper_mse": float(np.mean(diff[:, -1] ** 2)),
            "gripper_mae": float(np.mean(np.abs(diff[:, -1]))),
        }
    finally:
        if hasattr(env, "close"):
            env.close()

    if args.output_json is not None:
        out_path = Path(args.output_json).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
