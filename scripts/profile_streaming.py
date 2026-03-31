import json
import time
from pathlib import Path

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf, open_dict

from fastwam.runtime import build_datasets
from fastwam.utils.config_resolvers import register_default_resolvers
from fastwam.utils import misc

register_default_resolvers()
REPO_ROOT = Path(__file__).resolve().parents[1]


def _to_dtype(mixed_precision: str) -> torch.dtype:
    key = str(mixed_precision).lower()
    if key == "fp16":
        return torch.float16
    if key == "bf16":
        return torch.bfloat16
    return torch.float32


def _sync(device: str) -> None:
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize(torch.device(device))


def _measure(fn, warmup_iters: int, measure_iters: int, device: str) -> tuple[float, list[float]]:
    for _ in range(warmup_iters):
        fn()
        _sync(device)

    samples_ms: list[float] = []
    for _ in range(measure_iters):
        t0 = time.perf_counter()
        fn()
        _sync(device)
        samples_ms.append((time.perf_counter() - t0) * 1000.0)
    return sum(samples_ms) / max(len(samples_ms), 1), samples_ms


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


@hydra.main(config_path="../configs", config_name="profile/fastwam_streaming", version_base="1.3")
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
    sample = dataset[0]
    image = sample["video"][:, 0].unsqueeze(0).to(device=model.device, dtype=model.torch_dtype)
    context = sample["context"].unsqueeze(0).to(device=model.device, dtype=model.torch_dtype)
    context_mask = sample["context_mask"].unsqueeze(0).to(device=model.device, dtype=torch.bool)
    proprio = None
    if sample.get("proprio") is not None:
        proprio = sample["proprio"][0].to(device=model.device, dtype=model.torch_dtype)

    action = sample["action"]
    num_frames = int(sample["video"].shape[1])
    action_horizon = int(action.shape[0] // max(num_frames - 1, 1))
    action_horizon = max(1, action_horizon)

    warmup_iters = int(cfg.profile.warmup_iters)
    measure_iters = int(cfg.profile.measure_iters)
    num_inference_steps = int(cfg.profile.num_inference_steps)
    tiled = bool(cfg.profile.tiled)
    obs_timestamp_ms = float(cfg.profile.obs_timestamp_ms)
    rand_device = str(cfg.profile.rand_device)
    frontier_schedule = cfg.profile.get("frontier_schedule")
    if frontier_schedule is not None:
        frontier_schedule = list(frontier_schedule)

    def profile_video_prefill():
        model.build_streaming_video_cache_from_input_image(
            input_image=image,
            context=context,
            context_mask=context_mask,
            tiled=tiled,
        )

    def profile_action_full():
        model.reset_streaming_state()
        model.infer_action_streaming(
            prompt=None,
            input_image=image,
            action_horizon=action_horizon,
            proprio=proprio,
            context=context,
            context_mask=context_mask,
            num_inference_steps=num_inference_steps,
            rand_device=rand_device,
            tiled=tiled,
            frontier_schedule=frontier_schedule,
            obs_timestamp_ms=obs_timestamp_ms,
        )

    def profile_action_step():
        model.reset_streaming_state()
        model.submit_observation(
            input_image=image,
            prompt=None,
            context=context,
            context_mask=context_mask,
            obs_timestamp_ms=obs_timestamp_ms,
            tiled=tiled,
        )
        model.advance_video_cache_frontier(max_layers=model.mot.num_layers)
        job = model.start_action_job(
            action_horizon=action_horizon,
            prompt=None,
            context=context,
            context_mask=context_mask,
            proprio=proprio,
            num_inference_steps=num_inference_steps,
            rand_device=rand_device,
        )
        model.step_action_job(job)

    video_prefill_ms, video_prefill_samples = _measure(
        profile_video_prefill, warmup_iters=warmup_iters, measure_iters=measure_iters, device=str(cfg.device)
    )
    action_full_ms, action_full_samples = _measure(
        profile_action_full, warmup_iters=warmup_iters, measure_iters=measure_iters, device=str(cfg.device)
    )
    action_step_ms, action_step_samples = _measure(
        profile_action_step, warmup_iters=warmup_iters, measure_iters=measure_iters, device=str(cfg.device)
    )

    payload = {
        "config": OmegaConf.to_container(cfg, resolve=True),
        "metrics_ms": {
            "video_single_frame_prefill_avg_ms": video_prefill_ms,
            "action_full_denoise_avg_ms": action_full_ms,
            "action_single_step_avg_ms": action_step_ms,
        },
        "samples_ms": {
            "video_single_frame_prefill_ms": video_prefill_samples,
            "action_full_denoise_ms": action_full_samples,
            "action_single_step_ms": action_step_samples,
        },
    }
    output_path = Path(str(cfg.output_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload["metrics_ms"], indent=2))


if __name__ == "__main__":
    main()
