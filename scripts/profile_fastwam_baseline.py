import json
import time
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


def _summarize(samples_ms: list[float]) -> dict[str, float]:
    if not samples_ms:
        return {"avg_ms": 0.0, "min_ms": 0.0, "max_ms": 0.0}
    return {
        "avg_ms": float(sum(samples_ms) / len(samples_ms)),
        "min_ms": float(min(samples_ms)),
        "max_ms": float(max(samples_ms)),
    }


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


@hydra.main(config_path="../configs", config_name="profile/fastwam_baseline", version_base="1.3")
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
    sample = dataset[0]
    image = sample["video"][:, 0].unsqueeze(0).to(device=model.device, dtype=model.torch_dtype)
    base_context = sample["context"].unsqueeze(0).to(device=model.device, dtype=model.torch_dtype)
    base_context_mask = sample["context_mask"].unsqueeze(0).to(device=model.device, dtype=torch.bool)
    proprio = None
    if sample.get("proprio") is not None:
        proprio = sample["proprio"][0].to(device=model.device, dtype=model.torch_dtype)

    context = base_context
    context_mask = base_context_mask
    if proprio is not None:
        context, context_mask = model._append_proprio_to_context(
            context=base_context,
            context_mask=base_context_mask,
            proprio=proprio.unsqueeze(0) if proprio.ndim == 1 else proprio,
        )

    action = sample["action"]
    num_frames = int(sample["video"].shape[1])
    action_horizon = int(action.shape[0] // max(num_frames - 1, 1))
    action_horizon = max(1, action_horizon)

    warmup_iters = int(cfg.profile.warmup_iters)
    measure_iters = int(cfg.profile.measure_iters)
    num_inference_steps = int(cfg.profile.num_inference_steps)
    tiled = bool(cfg.profile.tiled)
    rand_device = str(cfg.profile.rand_device)

    def _encode_first_frame_latents():
        return model._encode_input_image_latents_tensor(input_image=image, tiled=tiled)

    def _build_video_pre(first_frame_latents: torch.Tensor):
        timestep_video = torch.zeros(
            (first_frame_latents.shape[0],),
            dtype=first_frame_latents.dtype,
            device=model.device,
        )
        fuse_flag = bool(getattr(model.video_expert, "fuse_vae_embedding_in_latents", False))
        video_pre = model.video_expert.pre_dit(
            x=first_frame_latents,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=fuse_flag,
        )
        return video_pre

    def _build_prefill_payload():
        first_frame_latents = _encode_first_frame_latents()
        video_pre = _build_video_pre(first_frame_latents)
        video_seq_len = int(video_pre["tokens"].shape[1])
        attention_mask = model._build_mot_attention_mask(
            video_seq_len=video_seq_len,
            action_seq_len=action_horizon,
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_pre["tokens"].device,
        )
        video_kv_cache = model.mot.prefill_video_cache(
            video_tokens=video_pre["tokens"],
            video_freqs=video_pre["freqs"],
            video_t_mod=video_pre["t_mod"],
            video_context_payload={
                "context": video_pre["context"],
                "mask": video_pre["context_mask"],
            },
            video_attention_mask=attention_mask[:video_seq_len, :video_seq_len],
        )
        return {
            "first_frame_latents": first_frame_latents,
            "video_pre": video_pre,
            "video_kv_cache": video_kv_cache,
            "video_seq_len": video_seq_len,
            "attention_mask": attention_mask,
        }

    prebuilt_payload = _build_prefill_payload()
    latents_action_template = torch.randn(
        (1, action_horizon, model.action_expert.action_dim),
        device=rand_device,
        dtype=torch.float32,
    ).to(device=model.device, dtype=model.torch_dtype)
    timesteps, deltas = model.infer_action_scheduler.build_inference_schedule(
        num_inference_steps=num_inference_steps,
        device=model.device,
        dtype=latents_action_template.dtype,
    )
    step_t0 = timesteps[0].unsqueeze(0).to(device=model.device, dtype=latents_action_template.dtype)
    step_delta0 = deltas[0]

    def profile_video_encode_only():
        _encode_first_frame_latents()

    def profile_video_pre_dit_only():
        first_frame_latents = _encode_first_frame_latents()
        _build_video_pre(first_frame_latents)

    def profile_video_cache_prefill_only():
        video_pre = prebuilt_payload["video_pre"]
        model.mot.prefill_video_cache(
            video_tokens=video_pre["tokens"],
            video_freqs=video_pre["freqs"],
            video_t_mod=video_pre["t_mod"],
            video_context_payload={
                "context": video_pre["context"],
                "mask": video_pre["context_mask"],
            },
            video_attention_mask=prebuilt_payload["attention_mask"][
                :prebuilt_payload["video_seq_len"], :prebuilt_payload["video_seq_len"]
            ],
        )

    def profile_video_prefill():
        _build_prefill_payload()

    def profile_action_full():
        model.infer_action(
            prompt=None,
            input_image=image,
            action_horizon=action_horizon,
            proprio=proprio,
            context=context,
            context_mask=context_mask,
            num_inference_steps=num_inference_steps,
            rand_device=rand_device,
            tiled=tiled,
        )

    def profile_action_noise_pred_single_step():
        latents_action = latents_action_template.clone()
        model._predict_action_noise_with_cache(
            latents_action=latents_action,
            timestep_action=step_t0,
            context=context,
            context_mask=context_mask,
            video_kv_cache=prebuilt_payload["video_kv_cache"],
            attention_mask=prebuilt_payload["attention_mask"],
            video_seq_len=prebuilt_payload["video_seq_len"],
        )

    def profile_action_scheduler_single_step():
        latents_action = latents_action_template.clone()
        pred_action = torch.randn_like(latents_action)
        model.infer_action_scheduler.step(pred_action, step_delta0, latents_action)

    def profile_action_single_step_total():
        latents_action = latents_action_template.clone()
        pred_action = model._predict_action_noise_with_cache(
            latents_action=latents_action,
            timestep_action=step_t0,
            context=context,
            context_mask=context_mask,
            video_kv_cache=prebuilt_payload["video_kv_cache"],
            attention_mask=prebuilt_payload["attention_mask"],
            video_seq_len=prebuilt_payload["video_seq_len"],
        )
        model.infer_action_scheduler.step(pred_action, step_delta0, latents_action)

    video_encode_ms, video_encode_samples = _measure(
        profile_video_encode_only, warmup_iters=warmup_iters, measure_iters=measure_iters, device=str(cfg.device)
    )
    video_pre_dit_ms, video_pre_dit_samples = _measure(
        profile_video_pre_dit_only, warmup_iters=warmup_iters, measure_iters=measure_iters, device=str(cfg.device)
    )
    video_cache_prefill_ms, video_cache_prefill_samples = _measure(
        profile_video_cache_prefill_only, warmup_iters=warmup_iters, measure_iters=measure_iters, device=str(cfg.device)
    )
    video_prefill_ms, video_prefill_samples = _measure(
        profile_video_prefill, warmup_iters=warmup_iters, measure_iters=measure_iters, device=str(cfg.device)
    )
    action_full_ms, action_full_samples = _measure(
        profile_action_full, warmup_iters=warmup_iters, measure_iters=measure_iters, device=str(cfg.device)
    )
    action_noise_pred_ms, action_noise_pred_samples = _measure(
        profile_action_noise_pred_single_step,
        warmup_iters=warmup_iters,
        measure_iters=measure_iters,
        device=str(cfg.device),
    )
    action_scheduler_step_ms, action_scheduler_step_samples = _measure(
        profile_action_scheduler_single_step,
        warmup_iters=warmup_iters,
        measure_iters=measure_iters,
        device=str(cfg.device),
    )
    action_step_ms, action_step_samples = _measure(
        profile_action_single_step_total, warmup_iters=warmup_iters, measure_iters=measure_iters, device=str(cfg.device)
    )

    payload = {
        "config": OmegaConf.to_container(cfg, resolve=True),
        "metrics_ms": {
            "video_first_frame_vae_encode_avg_ms": video_encode_ms,
            "video_pre_dit_avg_ms": video_pre_dit_ms,
            "video_cache_prefill_only_avg_ms": video_cache_prefill_ms,
            "video_single_frame_prefill_avg_ms": video_prefill_ms,
            "action_full_denoise_avg_ms": action_full_ms,
            "action_noise_pred_single_step_avg_ms": action_noise_pred_ms,
            "action_scheduler_single_step_avg_ms": action_scheduler_step_ms,
            "action_single_step_total_avg_ms": action_step_ms,
        },
        "samples_ms": {
            "video_first_frame_vae_encode_ms": video_encode_samples,
            "video_pre_dit_ms": video_pre_dit_samples,
            "video_cache_prefill_only_ms": video_cache_prefill_samples,
            "video_single_frame_prefill_ms": video_prefill_samples,
            "action_full_denoise_ms": action_full_samples,
            "action_noise_pred_single_step_ms": action_noise_pred_samples,
            "action_scheduler_single_step_ms": action_scheduler_step_samples,
            "action_single_step_total_ms": action_step_samples,
        },
        "summary_ms": {
            "video_first_frame_vae_encode": _summarize(video_encode_samples),
            "video_pre_dit": _summarize(video_pre_dit_samples),
            "video_cache_prefill_only": _summarize(video_cache_prefill_samples),
            "video_single_frame_prefill": _summarize(video_prefill_samples),
            "action_full_denoise": _summarize(action_full_samples),
            "action_noise_pred_single_step": _summarize(action_noise_pred_samples),
            "action_scheduler_single_step": _summarize(action_scheduler_step_samples),
            "action_single_step_total": _summarize(action_step_samples),
        },
    }
    output_path = Path(str(cfg.output_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload["metrics_ms"], indent=2))


if __name__ == "__main__":
    main()
