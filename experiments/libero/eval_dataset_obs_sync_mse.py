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
from omegaconf import OmegaConf, open_dict

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from experiments.libero.build_xt_replay import (  # noqa: E402
    _mixed_precision_to_model_dtype,
    _normalize_mixed_precision,
    resolve_weight_checkpoint,
)
from experiments.libero.libero_utils import invert_gripper_action  # noqa: E402
from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT  # noqa: E402
from fastwam.utils.pytorch_utils import set_global_seed  # noqa: E402


def _register_resolver_if_needed(name: str, fn) -> None:
    if not OmegaConf.has_resolver(name):
        OmegaConf.register_new_resolver(name, fn)


_register_resolver_if_needed("eval", eval)
_register_resolver_if_needed("max", lambda x: max(x))
_register_resolver_if_needed("split", lambda s, idx: s.split("/")[int(idx)])

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run synchronous infer_action() on dataset observations and compare both the first served "
            "action and the full predicted chunk against dataset ground-truth actions."
        )
    )
    parser.add_argument("--task-config", default="libero_streaming_action_ft_2cam224_1e-4")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--dataset-index", type=int, default=0)
    parser.add_argument("--episode-index", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=160)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--mixed-precision", default="bf16")
    parser.add_argument("--num-inference-steps", type=int, default=None)
    parser.add_argument("--action-horizon", type=int, default=None)
    parser.add_argument(
        "--skip-initial-steps",
        type=int,
        default=0,
        help="Exclude the first N env steps from the secondary summary block.",
    )
    parser.add_argument("--rand-device", default=None)
    parser.add_argument("--tiled", action="store_true")
    return parser.parse_args()


def _compose_cfg(args: argparse.Namespace):
    overrides = [
        f"task={args.task_config}",
        f"ckpt={args.ckpt}",
        f"mixed_precision={args.mixed_precision}",
        f"seed={int(args.seed)}",
    ]
    if args.device is not None:
        overrides.append(f"EVALUATION.device={args.device}")
    if args.num_inference_steps is not None:
        overrides.append(f"EVALUATION.num_inference_steps={int(args.num_inference_steps)}")
    if args.action_horizon is not None:
        overrides.append(f"EVALUATION.action_horizon={int(args.action_horizon)}")
    if args.rand_device is not None:
        overrides.append(f"EVALUATION.rand_device={args.rand_device}")
    if bool(args.tiled):
        overrides.append("EVALUATION.tiled=true")

    config_dir = str((project_root / "configs").resolve())
    with initialize_config_dir(version_base="1.3", config_dir=config_dir):
        return compose(config_name="sim_libero", overrides=overrides)


def _build_model(cfg, *, ckpt: str, model_dtype: torch.dtype, device: str):
    model = instantiate(cfg.model, model_dtype=model_dtype, device=device)
    weight_ckpt = resolve_weight_checkpoint(ckpt)
    model.load_checkpoint(str(weight_ckpt))
    return model.to(device).eval()


def _resolve_action_horizon(cfg) -> int:
    horizon_cfg = cfg.EVALUATION.get("action_horizon", None)
    if horizon_cfg is not None:
        return int(horizon_cfg)
    if cfg.data.train.get("action_horizon", None) is not None:
        return int(cfg.data.train.action_horizon)
    return int(cfg.data.train.num_frames) - 1


def _array_stats(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "min": None, "max": None}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.shape[0]),
        "mean": float(np.mean(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def _slice_rows(rows: list[dict[str, Any]], start: int, end: int | None = None) -> list[dict[str, Any]]:
    start = max(0, int(start))
    if end is None:
        return rows[start:]
    return rows[start : max(start, int(end))]


def _build_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    chunk_mses = [float(row["chunk_mse"]) for row in rows]
    chunk_env_mses = [
        float(row["chunk_env_action_mse"])
        for row in rows
        if row.get("chunk_env_action_mse") is not None
    ]
    head = rows[: min(8, len(rows))]
    next8 = _slice_rows(rows, 8, 16)
    tail = _slice_rows(rows, 16, None)
    return {
        "chunk_mse": _array_stats(chunk_mses),
        "env_action_mse": _array_stats(chunk_env_mses),
        "head8_env_action_mse": _array_stats(
            [float(row["chunk_env_action_mse"]) for row in head if row.get("chunk_env_action_mse") is not None]
        ),
        "next8_env_action_mse": _array_stats(
            [float(row["chunk_env_action_mse"]) for row in next8 if row.get("chunk_env_action_mse") is not None]
        ),
        "tail_env_action_mse": _array_stats(
            [float(row["chunk_env_action_mse"]) for row in tail if row.get("chunk_env_action_mse") is not None]
        ),
        "chunk_offset_env_action_mse": _mean_per_position(rows, "chunk_env_action_mse_per_position"),
    }


def _episode_index_from_dataset_index(dataset, dataset_index: int) -> int:
    if not hasattr(dataset, "sample_index"):
        raise ValueError("Expected StreamingRobotEpisodeDataset with sample_index.")
    if not 0 <= int(dataset_index) < len(dataset.sample_index):
        raise IndexError(f"dataset_index={dataset_index} out of range [0, {len(dataset.sample_index) - 1}]")
    episode_idx, _trigger_obs_idx, _raw_action_start = dataset.sample_index[int(dataset_index)]
    return int(episode_idx)


def _episode_image(dataset, payload: dict[str, Any], raw_frame_index: int) -> torch.Tensor:
    image = dataset._query_episode_images(
        dataset=payload["dataset"],
        local_episode_idx=int(payload["local_episode_idx"]),
        raw_frame_indices=[int(raw_frame_index)],
    )
    if image.ndim != 4 or int(image.shape[0]) != 1:
        raise ValueError(f"Expected one processed image frame [1,C,H,W], got {tuple(image.shape)}")
    return image[0].detach().to(device="cpu", dtype=torch.float32)


def _episode_proprio(dataset, payload: dict[str, Any], raw_frame_index: int) -> torch.Tensor:
    action_raw = {
        key: tensor[int(raw_frame_index) : int(raw_frame_index) + 1]
        for key, tensor in payload["action_raw"].items()
    }
    state_raw = {
        key: tensor[int(raw_frame_index) : int(raw_frame_index) + 1]
        for key, tensor in payload["state_raw"].items()
    }
    _action, state = dataset._normalize_action_and_state(action_raw=action_raw, state_raw=state_raw)
    return state.squeeze(0).detach().to(device="cpu", dtype=torch.float32)


def _dataset_action_to_env_action(action: np.ndarray, *, binarize_gripper: bool) -> np.ndarray:
    converted = np.asarray(action, dtype=np.float32).copy()
    converted[..., -1] = converted[..., -1] * 2.0 - 1.0
    converted = invert_gripper_action(converted)
    if bool(binarize_gripper):
        converted[..., -1] = np.sign(converted[..., -1])
    return np.asarray(converted, dtype=np.float32)


def _normalized_action_to_env_action(
    action: torch.Tensor,
    *,
    processor,
    binarize_gripper: bool,
) -> np.ndarray:
    if action.ndim == 2:
        action = action.unsqueeze(0)
    if action.ndim != 3:
        raise ValueError(f"Expected normalized action [B, T, D], got {tuple(action.shape)}")
    action_meta = processor.shape_meta["action"]
    if len(action_meta) != 1:
        raise ValueError("Expected a single merged action key in processor.shape_meta['action'].")
    action_key = action_meta[0]["key"]
    normalizer = processor.normalizer.normalizers["action"][action_key]
    action_cpu = action.detach().to(dtype=torch.float32, device="cpu")
    env_action = normalizer.backward(action_cpu).numpy()[0]
    env_action[..., -1] = env_action[..., -1] * 2.0 - 1.0
    env_action = invert_gripper_action(env_action)
    if bool(binarize_gripper):
        env_action[..., -1] = np.sign(env_action[..., -1])
    return np.asarray(env_action, dtype=np.float32)


def _raw_action_chunk_to_normalized(
    raw_chunk: np.ndarray,
    *,
    processor,
) -> torch.Tensor:
    action_meta = processor.shape_meta["action"]
    if len(action_meta) != 1:
        raise ValueError("Expected a single merged action key in processor.shape_meta['action'].")
    action_key = action_meta[0]["key"]
    normalizer = processor.normalizer.normalizers["action"][action_key]
    raw_chunk_t = torch.from_numpy(np.asarray(raw_chunk, dtype=np.float32)).unsqueeze(0)
    return normalizer.forward(raw_chunk_t).squeeze(0).to(dtype=torch.float32, device="cpu")


def _masked_mse(pred: torch.Tensor, target: torch.Tensor, action_is_pad: torch.Tensor | None = None) -> float:
    pred_f = pred.detach().float()
    target_f = target.detach().float()
    err = (pred_f - target_f).pow(2)
    if action_is_pad is None:
        return float(err.mean().item())
    pad = action_is_pad.detach().to(device=err.device)
    if pad.ndim == 1:
        pad = pad.unsqueeze(0)
    valid = (~pad.bool()).to(dtype=err.dtype)
    while valid.ndim < err.ndim:
        valid = valid.unsqueeze(-1)
    denom = valid.sum() * err.shape[-1]
    if float(denom.item()) <= 0.0:
        return float(err.mean().item())
    return float((err * valid).sum().div(denom).item())


def _masked_mse_per_position(
    pred: torch.Tensor,
    target: torch.Tensor,
    action_is_pad: torch.Tensor | None = None,
) -> list[float | None]:
    pred_f = pred.detach().float()
    target_f = target.detach().float()
    err = (pred_f - target_f).pow(2).mean(dim=-1)
    if action_is_pad is None:
        return [float(v) for v in err.mean(dim=0).tolist()]
    pad = action_is_pad.detach().to(device=err.device)
    if pad.ndim == 1:
        pad = pad.unsqueeze(0)
    valid = (~pad.bool()).to(dtype=err.dtype)
    per_pos: list[float | None] = []
    for pos in range(int(err.shape[1])):
        denom = float(valid[:, pos].sum().item())
        if denom <= 0.0:
            per_pos.append(None)
        else:
            per_pos.append(float((err[:, pos] * valid[:, pos]).sum().div(valid[:, pos].sum()).item()))
    return per_pos


def _mean_per_position(rows: list[dict[str, Any]], key: str) -> list[float | None] | None:
    values = [row.get(key) for row in rows if row.get(key) is not None]
    if not values:
        return None
    length = len(values[0])
    out: list[float | None] = []
    for idx in range(length):
        bucket = [float(v[idx]) for v in values if v[idx] is not None]
        out.append(None if len(bucket) == 0 else float(sum(bucket) / float(len(bucket))))
    return out


def _build_padded_raw_chunk(raw_action: np.ndarray, *, env_step: int, action_horizon: int) -> tuple[np.ndarray, np.ndarray]:
    action_dim = int(raw_action.shape[1])
    target = np.zeros((int(action_horizon), action_dim), dtype=np.float32)
    pad = np.ones((int(action_horizon),), dtype=bool)
    end = min(int(env_step) + int(action_horizon), int(raw_action.shape[0]))
    valid = max(0, int(end - int(env_step)))
    if valid > 0:
        target[:valid] = raw_action[int(env_step) : int(end)]
        pad[:valid] = False
    return target, pad


@torch.no_grad()
def run_dataset_obs_sync_mse(args: argparse.Namespace) -> dict[str, Any]:
    set_global_seed(max(int(args.seed), 1), get_worker_init_fn=False)
    cfg = _compose_cfg(args)

    sample_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    with open_dict(sample_cfg.data.train):
        sample_cfg.data.train.pop("trajectory_replay_key", None)
    dataset = instantiate(sample_cfg.data.train)

    episode_index = (
        int(args.episode_index)
        if args.episode_index is not None
        else _episode_index_from_dataset_index(dataset, int(args.dataset_index))
    )
    payload = dataset._load_episode_cache(int(episode_index))
    raw_action = payload["action_raw"]["default"].detach().cpu().numpy().astype(np.float32)
    raw_num_steps = int(raw_action.shape[0])
    max_steps = min(max(int(args.max_steps), 1), raw_num_steps)

    mixed_precision = _normalize_mixed_precision(str(cfg.get("mixed_precision", args.mixed_precision)))
    model_dtype = _mixed_precision_to_model_dtype(mixed_precision)
    device = str(args.device or cfg.EVALUATION.get("device") or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    if device == "cuda":
        device = "cuda:0"
    model = _build_model(cfg, ckpt=str(args.ckpt), model_dtype=model_dtype, device=device)

    processor = dataset.lerobot_dataset.processor
    if processor is None:
        raise ValueError("Dataset processor is missing.")

    prompt = DEFAULT_PROMPT.format(task=str(payload["instruction"]))
    with torch.no_grad():
        context, context_mask = model.encode_prompt(prompt)
    context = context.to(device="cpu", dtype=model.torch_dtype)
    context_mask = context_mask.to(device="cpu", dtype=torch.bool)

    binarize_gripper = bool(cfg.EVALUATION.get("binarize_gripper", False))
    action_horizon = int(_resolve_action_horizon(cfg))
    num_inference_steps = int(cfg.EVALUATION.get("num_inference_steps", cfg.get("eval_num_inference_steps", 10)))
    rand_device = str(cfg.EVALUATION.get("rand_device", "cpu"))
    tiled = bool(cfg.EVALUATION.get("tiled", False) or bool(args.tiled))

    rows: list[dict[str, Any]] = []
    for env_step in range(max_steps):
        model.reset_streaming_state()
        image = _episode_image(dataset, payload, int(env_step))
        proprio = _episode_proprio(dataset, payload, int(env_step))
        infer_out = model.infer_action(
            prompt=None,
            input_image=image,
            action_horizon=action_horizon,
            proprio=proprio,
            context=context,
            context_mask=context_mask,
            num_inference_steps=int(num_inference_steps),
            seed=int(args.seed) + int(env_step),
            rand_device=str(rand_device),
            tiled=bool(tiled),
        )

        pred_action = infer_out["action"].detach().to(device="cpu", dtype=torch.float32)
        target_raw_chunk, action_is_pad_np = _build_padded_raw_chunk(
            raw_action,
            env_step=int(env_step),
            action_horizon=int(action_horizon),
        )
        target_action = _raw_action_chunk_to_normalized(target_raw_chunk, processor=processor)
        action_is_pad = torch.from_numpy(action_is_pad_np)

        pred_env_chunk = _normalized_action_to_env_action(
            pred_action,
            processor=processor,
            binarize_gripper=binarize_gripper,
        )
        target_env_chunk = _dataset_action_to_env_action(
            target_raw_chunk,
            binarize_gripper=binarize_gripper,
        )

        first_pred = np.asarray(pred_env_chunk[0], dtype=np.float32)
        first_target = np.asarray(target_env_chunk[0], dtype=np.float32)
        first_diff = first_pred - first_target

        chunk_mse = _masked_mse(
            pred_action.unsqueeze(0),
            target_action.unsqueeze(0),
            action_is_pad.unsqueeze(0),
        )
        chunk_mse_per_position = _masked_mse_per_position(
            pred_action.unsqueeze(0),
            target_action.unsqueeze(0),
            action_is_pad.unsqueeze(0),
        )

        valid_mask = ~action_is_pad_np
        if bool(np.any(valid_mask)):
            chunk_env_sqerr = np.square(pred_env_chunk[valid_mask] - target_env_chunk[valid_mask])
            chunk_env_mse = float(np.mean(chunk_env_sqerr))
            chunk_env_mse_per_position = [
                (None if bool(action_is_pad_np[pos]) else float(np.mean(np.square(pred_env_chunk[pos] - target_env_chunk[pos]))))
                for pos in range(int(action_horizon))
            ]
        else:
            chunk_env_mse = None
            chunk_env_mse_per_position = [None] * int(action_horizon)

        rows.append(
            {
                "env_step": int(env_step),
                "first_env_action_mse": float(np.mean(np.square(first_diff))),
                "first_env_action_mae": float(np.mean(np.abs(first_diff))),
                "chunk_mse": float(chunk_mse),
                "chunk_rmse": float(chunk_mse ** 0.5),
                "chunk_mse_per_position": chunk_mse_per_position,
                "chunk_env_action_mse": chunk_env_mse,
                "chunk_env_action_mse_per_position": chunk_env_mse_per_position,
                "first_pred_action": first_pred.astype(float).tolist(),
                "first_target_action": first_target.astype(float).tolist(),
            }
        )

    skip_initial_steps = max(0, int(args.skip_initial_steps))
    rows_after_skip = rows[skip_initial_steps:]

    result = {
        "ckpt": str(resolve_weight_checkpoint(str(args.ckpt))),
        "task_config": str(args.task_config),
        "dataset_index": int(args.dataset_index),
        "episode_index": int(episode_index),
        "instruction": str(payload["instruction"]),
        "seed": int(args.seed),
        "device": str(device),
        "mixed_precision": str(mixed_precision),
        "num_episode_raw_actions": int(raw_num_steps),
        "num_compared_steps": int(len(rows)),
        "action_horizon": int(action_horizon),
        "num_inference_steps": int(num_inference_steps),
        "tiled": bool(tiled),
        "summary": _build_summary(rows),
        "summary_after_skip": {
            "skip_initial_steps": int(skip_initial_steps),
            **_build_summary(rows_after_skip),
        },
        "per_step": rows,
    }
    return result


def main() -> None:
    args = _parse_args()
    result = run_dataset_obs_sync_mse(args)
    output_path = Path(args.output_json).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, cls=NumpyEncoder)
    print(f"Wrote dataset-observation sync MSE to: {output_path}")
    print(json.dumps(result["summary"], indent=2, cls=NumpyEncoder))


if __name__ == "__main__":
    main()
