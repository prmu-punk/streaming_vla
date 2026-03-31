import json
from pathlib import Path

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, open_dict

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


@hydra.main(config_path="../configs", config_name="profile/fastwam_async_runtime_trace", version_base="1.3")
def main(cfg: DictConfig):
    cfg.data = _absolutize_data_cfg(cfg.data)
    cfg.data = _maybe_attach_default_norm_stats(cfg.data)
    misc.register_work_dir(str((REPO_ROOT / "runs/profile").resolve()))

    model_dtype = _to_dtype(cfg.mixed_precision)
    model = instantiate(cfg.model, model_dtype=model_dtype, device=str(cfg.device))
    if cfg.get("checkpoint_path"):
        model.load_checkpoint(str(cfg.checkpoint_path))
    model.eval()

    dataset, _ = build_datasets(cfg.data)
    sample = dataset[int(cfg.trace.sample_index)]
    video = sample["video"].to(device=model.device, dtype=model.torch_dtype)
    context = sample["context"].unsqueeze(0).to(device=model.device, dtype=model.torch_dtype)
    context_mask = sample["context_mask"].unsqueeze(0).to(device=model.device, dtype=torch.bool)

    obs_start_index = int(cfg.trace.obs_start_index)
    num_obs = int(cfg.trace.num_obs)
    obs_end_index = obs_start_index + num_obs
    if obs_start_index < 0 or obs_end_index > int(video.shape[1]):
        raise ValueError(
            f"Requested obs window [{obs_start_index}, {obs_end_index}) exceeds num_video_frames={int(video.shape[1])}."
        )
    obs_images = video[:, obs_start_index:obs_end_index].permute(1, 0, 2, 3).contiguous()
    obs_indices = list(range(obs_start_index, obs_end_index))

    proprio_seq = None
    if sample.get("proprio") is not None:
        proprio_seq = sample["proprio"][obs_start_index:obs_end_index].to(device=model.device, dtype=model.torch_dtype)

    action = sample["action"]
    num_frames = int(sample["video"].shape[1])
    action_horizon = int(action.shape[0] // max(num_frames - 1, 1))
    action_horizon = max(1, action_horizon)

    payload = model.simulate_async_runtime_trace(
        observation_images=obs_images,
        action_horizon=action_horizon,
        action_trigger_every_n_obs=int(cfg.trace.action_trigger_every_n_obs),
        obs_dt_ms=float(cfg.trace.obs_dt_ms),
        prompt=None,
        context=context,
        context_mask=context_mask,
        proprio_seq=proprio_seq,
        obs_indices=obs_indices,
        sigma_shift=(None if cfg.trace.get("sigma_shift") is None else float(cfg.trace.sigma_shift)),
        seed=(None if cfg.trace.get("seed") is None else int(cfg.trace.seed)),
        rand_device=str(cfg.trace.rand_device),
        tiled=bool(cfg.trace.tiled),
        warmup_video_bootstrap=bool(cfg.trace.get("warmup_video_bootstrap", True)),
        warmup_action_job=bool(cfg.trace.get("warmup_action_job", True)),
    )

    for job in payload["jobs"]:
        for step in job["steps"]:
            step["layer_sources"] = _compress_layer_sources(
                step["layer_obs_indices"],
                step["layer_obs_timestamps_ms"],
            )
            if not bool(cfg.trace.get("include_layer_arrays", False)):
                step.pop("layer_version_ids", None)
                step.pop("layer_obs_indices", None)
                step.pop("layer_obs_timestamps_ms", None)

    payload.update(
        {
            "sample_index": int(cfg.trace.sample_index),
            "obs_start_index": int(obs_start_index),
            "num_obs": int(num_obs),
        }
    )
    output_path = Path(str(cfg.output_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
