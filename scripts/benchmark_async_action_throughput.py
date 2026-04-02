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
    if default_stats_path is not None:
        with open_dict(train_cfg):
            train_cfg.pretrained_norm_stats = str(default_stats_path)
    return data_cfg


def _count_subsampled_obs(num_env_steps: int, obs_stride_env_steps: int) -> int:
    if num_env_steps <= 0:
        return 0
    return ((int(num_env_steps) - 1) // int(obs_stride_env_steps)) + 1


@hydra.main(config_path="../configs", config_name="profile/fastwam_async_action_throughput", version_base="1.3")
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
    sample = dataset[int(cfg.benchmark.sample_index)]
    video = sample["video"].to(device=model.device, dtype=model.torch_dtype)
    context = sample["context"].unsqueeze(0).to(device=model.device, dtype=model.torch_dtype)
    context_mask = sample["context_mask"].unsqueeze(0).to(device=model.device, dtype=torch.bool)

    num_video_frames = int(video.shape[1])
    warmup_env_steps = int(cfg.benchmark.warmup_env_steps)
    total_env_steps = int(cfg.benchmark.total_env_steps)
    obs_stride_env_steps = int(cfg.benchmark.obs_stride_env_steps)
    control_dt_ms = float(cfg.benchmark.control_dt_ms)
    action_trigger_every_n_obs = int(cfg.benchmark.action_trigger_every_n_obs)
    obs_dt_ms = control_dt_ms * float(obs_stride_env_steps)

    proprio_seq = None
    if sample.get("proprio") is not None:
        proprio_all = sample["proprio"].to(device=model.device, dtype=model.torch_dtype)
        if proprio_all.ndim == 2 and proprio_all.shape[0] > 0:
            actions_per_transition = max(int(sample["action"].shape[0] // max(num_video_frames - 1, 1)), 1)
            proprio_seq = []
            for frame_idx in range(num_video_frames):
                raw_idx = min(frame_idx * actions_per_transition, proprio_all.shape[0] - 1)
                proprio_seq.append(proprio_all[raw_idx])
            proprio_seq = torch.stack(proprio_seq, dim=0)

    action_horizon = int(cfg.benchmark.action_horizon)
    if action_horizon <= 0:
        actions_per_transition = max(int(sample["action"].shape[0] // max(num_video_frames - 1, 1)), 1)
        action_horizon = max(1, actions_per_transition)
    replan_steps = int(cfg.benchmark.get("replan_steps", action_horizon))
    if replan_steps <= 0:
        raise ValueError(f"`benchmark.replan_steps` must be positive, got {replan_steps}.")
    executed_horizon = min(action_horizon, replan_steps)
    warmup_obs_count = _count_subsampled_obs(warmup_env_steps, obs_stride_env_steps)
    formal_obs_count = _count_subsampled_obs(total_env_steps, obs_stride_env_steps)
    total_obs_count = warmup_obs_count + formal_obs_count

    frame_ids = [idx % num_video_frames for idx in range(total_obs_count)]
    observation_images = video[:, frame_ids].permute(1, 0, 2, 3).contiguous()
    repeated_proprio_seq = None
    if proprio_seq is not None:
        repeated_proprio_seq = proprio_seq[frame_ids].contiguous()

    trace_payload = model.simulate_async_runtime_trace(
        observation_images=observation_images,
        action_horizon=action_horizon,
        action_trigger_every_n_obs=action_trigger_every_n_obs,
        obs_dt_ms=obs_dt_ms,
        prompt=None,
        context=context,
        context_mask=context_mask,
        proprio_seq=repeated_proprio_seq,
        obs_indices=list(range(total_obs_count)),
        num_inference_steps=int(cfg.benchmark.num_inference_steps),
        sigma_shift=(None if cfg.benchmark.get("sigma_shift") is None else float(cfg.benchmark.sigma_shift)),
        seed=(None if cfg.benchmark.get("seed") is None else int(cfg.benchmark.seed)),
        rand_device=str(cfg.benchmark.rand_device),
        tiled=bool(cfg.benchmark.tiled),
        warmup_video_bootstrap=True,
        warmup_action_job=False,
        warmup_num_obs=warmup_obs_count,
        video_layers_per_chunk=int(cfg.benchmark.video_layers_per_chunk),
    )

    jobs = list(trace_payload["jobs"])
    completed_jobs = int(len(jobs))
    total_chunk_action_steps = int(completed_jobs * executed_horizon)
    planned_action_timestamps: set[int] = set()
    for job in jobs:
        trigger_obs_index = int(job["trigger_obs_index"])
        trigger_env_step = int((trigger_obs_index - warmup_obs_count) * obs_stride_env_steps)
        for i in range(executed_horizon):
            planned_action_timestamps.add(trigger_env_step + i)
    unique_planned_action_steps = int(len(planned_action_timestamps))
    overlapped_planned_action_steps = int(total_chunk_action_steps - unique_planned_action_steps)
    last_job_end_ms = max((float(job["job_wall_end_ms"]) for job in jobs), default=0.0)
    formal_submit_end_ms = float(max(formal_obs_count - 1, 0)) * obs_dt_ms
    elapsed_s = max(last_job_end_ms, formal_submit_end_ms) / 1000.0
    job_durations_ms = [float(job["job_wall_duration_ms"]) for job in jobs]
    if job_durations_ms:
        durations = torch.tensor(job_durations_ms, dtype=torch.float32)
        job_duration_avg_ms = float(durations.mean().item())
        job_duration_min_ms = float(durations.min().item())
        job_duration_max_ms = float(durations.max().item())
        job_duration_p50_ms = float(torch.quantile(durations, 0.50).item())
        job_duration_p90_ms = float(torch.quantile(durations, 0.90).item())
    else:
        job_duration_avg_ms = 0.0
        job_duration_min_ms = 0.0
        job_duration_max_ms = 0.0
        job_duration_p50_ms = 0.0
        job_duration_p90_ms = 0.0
    payload = {
        "control_dt_ms": control_dt_ms,
        "warmup_env_steps": warmup_env_steps,
        "total_env_steps": total_env_steps,
        "obs_stride_env_steps": obs_stride_env_steps,
        "obs_dt_ms": float(obs_dt_ms),
        "warmup_obs_count": int(warmup_obs_count),
        "formal_obs_count": int(formal_obs_count),
        "action_trigger_every_n_obs": action_trigger_every_n_obs,
        "action_horizon": int(action_horizon),
        "replan_steps": int(replan_steps),
        "executed_horizon": int(executed_horizon),
        "video_layers_per_chunk": int(cfg.benchmark.video_layers_per_chunk),
        "completed_jobs": int(completed_jobs),
        "completed_jobs_per_sec": float(completed_jobs / max(elapsed_s, 1e-6)),
        "total_chunk_action_steps": int(total_chunk_action_steps),
        "chunk_action_steps_per_sec": float(total_chunk_action_steps / max(elapsed_s, 1e-6)),
        "unique_planned_action_steps": int(unique_planned_action_steps),
        "unique_planned_action_steps_per_sec": float(unique_planned_action_steps / max(elapsed_s, 1e-6)),
        "overlapped_planned_action_steps": int(overlapped_planned_action_steps),
        "overlap_fraction": float(overlapped_planned_action_steps / max(total_chunk_action_steps, 1)),
        "control_steps_per_sec": float(total_env_steps / max(elapsed_s, 1e-6)),
        "realtime_factor": float((total_env_steps * control_dt_ms / 1000.0) / max(elapsed_s, 1e-6)),
        "trace_elapsed_ms": float(max(last_job_end_ms, formal_submit_end_ms)),
        "job_duration_avg_ms": float(job_duration_avg_ms),
        "job_duration_min_ms": float(job_duration_min_ms),
        "job_duration_max_ms": float(job_duration_max_ms),
        "job_duration_p50_ms": float(job_duration_p50_ms),
        "job_duration_p90_ms": float(job_duration_p90_ms),
        "sample_index": int(cfg.benchmark.sample_index),
    }
    output_path = Path(str(cfg.output_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
