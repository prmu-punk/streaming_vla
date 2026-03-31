import json
from pathlib import Path

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf, open_dict

from fastwam.runtime import build_datasets
from fastwam.utils import misc
from fastwam.utils.config_resolvers import register_default_resolvers

register_default_resolvers()
REPO_ROOT = Path(__file__).resolve().parents[1]


def _to_dtype(mixed_precision: str) -> torch.dtype:
    key = str(mixed_precision).lower()
    if key == "fp16":
        return torch.float16
    if key == "bf16":
        return torch.bfloat16
    return torch.float32


def _resolve_repo_path(path_like: str | None) -> str | None:
    if path_like is None:
        return None
    path = Path(str(path_like))
    if path.is_absolute():
        return str(path)
    return str((REPO_ROOT / path).resolve())


def _absolutize_data_cfg(data_cfg: DictConfig) -> DictConfig:
    for split_name in ["train", "val"]:
        split_cfg = data_cfg.get(split_name)
        if split_cfg is None:
            continue
        dataset_dirs = split_cfg.get("dataset_dirs")
        if dataset_dirs is not None:
            split_cfg.dataset_dirs = [_resolve_repo_path(v) for v in dataset_dirs]
        if split_cfg.get("text_embedding_cache_dir") is not None:
            split_cfg.text_embedding_cache_dir = _resolve_repo_path(split_cfg.text_embedding_cache_dir)
        if split_cfg.get("pretrained_norm_stats") is not None:
            split_cfg.pretrained_norm_stats = _resolve_repo_path(split_cfg.pretrained_norm_stats)
    return data_cfg


def _maybe_attach_default_norm_stats(data_cfg: DictConfig) -> DictConfig:
    train_cfg = data_cfg.get("train")
    if train_cfg is None or train_cfg.get("pretrained_norm_stats") is not None:
        return data_cfg
    dataset_dirs = [str(v) for v in train_cfg.get("dataset_dirs", [])]
    default_stats_path = None
    if any("libero" in v for v in dataset_dirs):
        candidate = REPO_ROOT / "checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json"
        if candidate.exists():
            default_stats_path = candidate
    elif any("robotwin" in v for v in dataset_dirs):
        candidate = REPO_ROOT / "checkpoints/fastwam_release/robotwin_uncond_3cam_384_dataset_stats.json"
        if candidate.exists():
            default_stats_path = candidate
    if default_stats_path is not None:
        with open_dict(train_cfg):
            train_cfg.pretrained_norm_stats = str(default_stats_path)
    return data_cfg


def _compress_layer_sources(layer_obs_indices: list[int], layer_obs_timestamps_ms: list[float]) -> list[dict]:
    if not layer_obs_indices:
        return []
    runs: list[dict] = []
    start = 0
    current_idx = int(layer_obs_indices[0])
    current_ts = float(layer_obs_timestamps_ms[0])
    for i in range(1, len(layer_obs_indices) + 1):
        boundary = i == len(layer_obs_indices)
        if boundary or int(layer_obs_indices[i]) != current_idx or float(layer_obs_timestamps_ms[i]) != current_ts:
            runs.append(
                {
                    "layer_start": int(start),
                    "layer_end_exclusive": int(i),
                    "obs_index": int(current_idx),
                    "obs_timestamp_ms": float(current_ts),
                }
            )
            if not boundary:
                start = i
                current_idx = int(layer_obs_indices[i])
                current_ts = float(layer_obs_timestamps_ms[i])
    return runs


def _default_frontier_targets(num_layers: int, num_steps: int) -> list[int]:
    if num_steps <= 1:
        return [num_layers]
    return [int(round(num_layers * step / (num_steps - 1))) for step in range(num_steps)]


@hydra.main(config_path="../configs", config_name="profile/fastwam_streaming_trace", version_base="1.3")
def main(cfg: DictConfig):
    cfg.data = _absolutize_data_cfg(cfg.data)
    cfg.data = _maybe_attach_default_norm_stats(cfg.data)
    misc.register_work_dir(str((REPO_ROOT / "runs/profile").resolve()))

    model_dtype = _to_dtype(cfg.mixed_precision)
    model = instantiate(cfg.model, model_dtype=model_dtype, device=str(cfg.device))
    if cfg.get("checkpoint_path"):
        model.load_checkpoint(str(cfg.checkpoint_path))
    model.eval()
    model.reset_streaming_state()

    dataset, _ = build_datasets(cfg.data)
    sample = dataset[int(cfg.trace.sample_index)]
    video = sample["video"].to(device=model.device, dtype=model.torch_dtype)
    context = sample["context"].unsqueeze(0).to(device=model.device, dtype=model.torch_dtype)
    context_mask = sample["context_mask"].unsqueeze(0).to(device=model.device, dtype=torch.bool)

    current_obs_index = int(cfg.trace.current_obs_index)
    obs_gap = int(cfg.trace.obs_gap)
    previous_obs_index = current_obs_index - obs_gap
    if previous_obs_index < 0:
        raise ValueError(f"`current_obs_index - obs_gap` must be >= 0, got {current_obs_index} - {obs_gap}.")
    if current_obs_index >= int(video.shape[1]):
        raise ValueError(
            f"`current_obs_index` must be < num_video_frames ({int(video.shape[1])}), got {current_obs_index}."
        )

    previous_image = video[:, previous_obs_index].unsqueeze(0)
    current_image = video[:, current_obs_index].unsqueeze(0)

    proprio = None
    if sample.get("proprio") is not None:
        proprio_seq = sample["proprio"].to(device=model.device, dtype=model.torch_dtype)
        proprio_idx = min(current_obs_index, int(proprio_seq.shape[0]) - 1)
        proprio = proprio_seq[proprio_idx]

    action = sample["action"]
    num_frames = int(sample["video"].shape[1])
    action_horizon = int(action.shape[0] // max(num_frames - 1, 1))
    action_horizon = max(1, action_horizon)
    num_inference_steps = int(cfg.trace.num_inference_steps)
    obs_dt_ms = float(cfg.trace.obs_dt_ms)

    frontier_targets = cfg.trace.get("frontier_targets")
    if frontier_targets is None:
        frontier_targets = _default_frontier_targets(model.mot.num_layers, num_inference_steps)
    frontier_targets = [int(v) for v in frontier_targets]
    if len(frontier_targets) != num_inference_steps:
        raise ValueError(
            f"`frontier_targets` length must equal num_inference_steps ({num_inference_steps}), got {len(frontier_targets)}."
        )

    model.submit_observation(
        input_image=previous_image,
        prompt=None,
        context=context,
        context_mask=context_mask,
        obs_index=previous_obs_index,
        obs_timestamp_ms=float(previous_obs_index) * obs_dt_ms,
        tiled=bool(cfg.trace.tiled),
    )
    model.submit_observation(
        input_image=current_image,
        prompt=None,
        context=context,
        context_mask=context_mask,
        obs_index=current_obs_index,
        obs_timestamp_ms=float(current_obs_index) * obs_dt_ms,
        tiled=bool(cfg.trace.tiled),
    )

    job = model.start_action_job(
        action_horizon=action_horizon,
        prompt=None,
        context=context,
        context_mask=context_mask,
        proprio=proprio,
        num_inference_steps=num_inference_steps,
        seed=(None if cfg.trace.get("seed") is None else int(cfg.trace.seed)),
        rand_device=str(cfg.trace.rand_device),
    )

    trace_steps: list[dict] = []
    current_frontier = 0
    for step_idx, target_frontier in enumerate(frontier_targets):
        if target_frontier < current_frontier:
            raise ValueError(
                f"`frontier_targets` must be monotonic non-decreasing, got {target_frontier} after {current_frontier}."
            )
        delta = target_frontier - current_frontier
        if delta > 0:
            model.advance_video_cache_frontier(max_layers=delta)
        current_frontier = target_frontier
        snapshot = model.snapshot_cache_for_action_step()
        trace_steps.append(
            {
                "denoise_step": int(step_idx),
                "frontier": int(snapshot.frontier),
                "snapshot_version": int(snapshot.version),
                "snapshot_obs_index": int(snapshot.obs_index),
                "snapshot_obs_timestamp_ms": float(snapshot.obs_timestamp_ms),
                "layer_sources": _compress_layer_sources(
                    snapshot.layer_obs_indices,
                    snapshot.layer_obs_timestamps_ms,
                ),
            }
        )
        model.step_action_job(job, snapshot=snapshot)

    payload = {
        "sample_index": int(cfg.trace.sample_index),
        "previous_obs_index": int(previous_obs_index),
        "current_obs_index": int(current_obs_index),
        "obs_gap": int(obs_gap),
        "obs_dt_ms": float(obs_dt_ms),
        "num_inference_steps": int(num_inference_steps),
        "frontier_targets": frontier_targets,
        "trace": trace_steps,
    }
    output_path = Path(str(cfg.output_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
