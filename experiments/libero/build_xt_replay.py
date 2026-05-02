from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf, open_dict

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from fastwam.models.wan22.streaming_cache import CacheSnapshot  # noqa: E402
from fastwam.utils.pytorch_utils import set_global_seed  # noqa: E402


def _register_resolver_if_needed(name: str, fn) -> None:
    if not OmegaConf.has_resolver(name):
        OmegaConf.register_new_resolver(name, fn)


_register_resolver_if_needed("eval", eval)
_register_resolver_if_needed("max", lambda x: max(x))
_register_resolver_if_needed("split", lambda s, idx: s.split("/")[int(idx)])


def _normalize_mixed_precision(mixed_precision: str) -> str:
    key = str(mixed_precision).strip().lower()
    if key not in {"no", "fp16", "bf16"}:
        raise ValueError(
            f"Unsupported mixed_precision: {mixed_precision}. "
            "Expected one of: ['no', 'fp16', 'bf16']."
        )
    return key


def _mixed_precision_to_model_dtype(mixed_precision: str) -> torch.dtype:
    precision = _normalize_mixed_precision(mixed_precision)
    if precision == "no":
        return torch.float32
    if precision == "fp16":
        return torch.float16
    return torch.bfloat16


def resolve_weight_checkpoint(path: str | Path) -> Path:
    path = Path(path)
    if path.is_file():
        return path
    if path.is_dir() and path.name.startswith("step_") and path.parent.name == "state":
        weight_path = path.parent.parent / "weights" / f"{path.name}.pt"
        if weight_path.exists():
            return weight_path
    raise FileNotFoundError(f"Cannot resolve model weight checkpoint from: {path}")


def _batched(sample: dict[str, Any]) -> dict[str, Any]:
    out = dict(sample)
    for key in ("obs_prev", "obs_cur", "obs_next", "obs_next2"):
        if isinstance(out.get(key), torch.Tensor) and out[key].ndim == 3:
            out[key] = out[key].unsqueeze(0)
    for key in ("target_action",):
        if isinstance(out.get(key), torch.Tensor) and out[key].ndim == 2:
            out[key] = out[key].unsqueeze(0)
    for key in ("action_is_pad", "proprio_t", "context_mask"):
        if isinstance(out.get(key), torch.Tensor) and out[key].ndim == 1:
            out[key] = out[key].unsqueeze(0)
    if isinstance(out.get("context"), torch.Tensor) and out["context"].ndim == 2:
        out["context"] = out["context"].unsqueeze(0)
    return out


def _load_schedule_pool(schedule_path: str | Path) -> list[dict[str, Any]]:
    payload = torch.load(str(schedule_path), map_location="cpu")
    schedules = payload.get("schedules", payload) if isinstance(payload, dict) else payload
    if not isinstance(schedules, list) or len(schedules) == 0:
        raise ValueError(f"Schedule file has no schedule traces: {schedule_path}")
    kept = []
    for schedule in schedules:
        steps = sorted(list(schedule.get("steps", [])), key=lambda row: int(row["denoise_step"]))
        if len(steps) == 0:
            continue
        row = dict(schedule)
        row["steps"] = steps
        kept.append(row)
    if len(kept) == 0:
        raise ValueError(f"Schedule file has no non-empty schedule traces: {schedule_path}")
    return kept


def _layer_keys_from_schedule(model, *, step: dict[str, Any], trigger_obs_index: int) -> tuple[list[str], list[int]]:
    layer_obs_indices = [int(v) for v in list(step["layer_obs_indices"])]
    raw_source_offsets = [int(v) - int(trigger_obs_index) for v in layer_obs_indices]
    layer_keys = [model._cache_key_from_offset(offset) for offset in raw_source_offsets]
    source_offsets = [model._cache_key_to_source_delta(key) for key in layer_keys]
    return layer_keys, source_offsets


@torch.no_grad()
def _sample_one(
    model,
    sample: dict[str, Any],
    dataset_index: int,
    *,
    schedule: dict[str, Any],
    schedule_index: int,
    num_inference_steps: int,
    seed: int,
    rand_device: str,
    xt_dtype: torch.dtype,
) -> list[dict[str, Any]]:
    batch = model._extract_streaming_episode_batch(_batched(sample))
    resolved_context, resolved_context_mask = model._resolve_streaming_condition_inputs(
        prompt=None,
        context=batch["context"],
        context_mask=batch["context_mask"],
        proprio=batch["proprio_t"],
    )
    batch = dict(batch)
    batch["context"] = resolved_context
    batch["context_mask"] = resolved_context_mask
    action_horizon = int(batch["target_action"].shape[1])
    job = model.start_action_job(
        action_horizon=action_horizon,
        context=batch["context"],
        context_mask=batch["context_mask"],
        trigger_obs_index=0,
        num_inference_steps=int(num_inference_steps),
        seed=int(seed),
        rand_device=str(rand_device),
    )
    caches = model._build_selected_video_cache_payload(batch, required_cache_keys=["prev", "cur", "next", "next2"])
    records: list[dict[str, Any]] = []
    schedule_steps = list(schedule["steps"])
    profile_trigger_obs_index = int(schedule.get("trigger_obs_index", 0))

    while not job.done:
        step_idx = int(job.current_step_idx)
        if step_idx >= len(schedule_steps):
            break
        step = schedule_steps[step_idx]
        if int(step["denoise_step"]) != step_idx:
            raise ValueError(
                f"Schedule {schedule_index} denoise step mismatch: expected {step_idx}, got {step['denoise_step']}."
            )
        layer_keys, source_offsets = _layer_keys_from_schedule(
            model,
            step=step,
            trigger_obs_index=profile_trigger_obs_index,
        )
        if len(layer_keys) != int(model.mot.num_layers):
            raise ValueError(
                f"Schedule {schedule_index} step {step_idx} has {len(layer_keys)} layers, expected {model.mot.num_layers}."
            )
        cache_layers = model._compose_replay_layer_cache(
            caches=caches,
            layer_cache_keys=[",".join(layer_keys)],
        )
        snapshot = CacheSnapshot(
            version=step_idx,
            obs_timestamp_ms=0.0,
            frontier=int(step.get("frontier", model.mot.num_layers)),
            video_seq_len=int(caches["video_seq_len"]),
            tokens_per_frame=int(caches["tokens_per_frame"]),
            cache_layers=cache_layers,
            context=batch["context"],
            context_mask=batch["context_mask"],
            obs_index=int(max(source_offsets)),
            layer_version_ids=[step_idx] * int(model.mot.num_layers),
            layer_obs_indices=source_offsets,
            layer_obs_timestamps_ms=[0.0] * int(model.mot.num_layers),
            layer_ready_events=[None] * int(model.mot.num_layers),
        )
        records.append(
            {
                "dataset_index": int(dataset_index),
                "episode_index": int(sample["episode_idx"]),
                "env_step": int(sample["raw_action_start"]),
                "denoise_step": step_idx,
                "timestep": float(job.timesteps[step_idx].detach().float().item()),
                "mode": str(step.get("mode", "")),
                "frontier": int(step.get("frontier", model.mot.num_layers)),
                "layer_cache_keys": layer_keys,
                "layer_obs_indices": source_offsets,
                "schedule_index": int(schedule_index),
                "x_t": job.latents_action[0].detach().cpu().to(dtype=xt_dtype),
            }
        )
        model.step_action_job(job, snapshot=snapshot)
    return records


def collect_xt_replay_from_schedule(
    cfg: DictConfig,
    *,
    ckpt: str,
    schedule_path: str | Path,
    output_dir: str | Path,
    max_samples: int,
    seed: int = 1,
    device: str | None = None,
    rand_device: str = "cpu",
    max_records: int | None = None,
    sample_start: int = 0,
    sample_stride: int = 1,
    save_to_disk: bool = True,
) -> list[dict[str, Any]]:
    seed = max(1, int(seed))
    set_global_seed(seed)
    schedules = _load_schedule_pool(schedule_path)
    mixed_precision = _normalize_mixed_precision(str(cfg.mixed_precision))
    model_dtype = _mixed_precision_to_model_dtype(mixed_precision)
    model_device = str(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    model = instantiate(cfg.model, model_dtype=model_dtype, device=model_device)
    weight_ckpt = resolve_weight_checkpoint(ckpt)
    model.load_checkpoint(str(weight_ckpt))
    model.eval()

    sample_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    with open_dict(sample_cfg.data.train):
        sample_cfg.data.train.pop("trajectory_replay_key", None)
    dataset = instantiate(sample_cfg.data.train)
    num_steps = int(cfg.model.streaming.streaming_train.get("infer_num_inference_steps", 10))
    replay_multiplier = max(int(cfg.xt_replay.get("replay_multiplier", 4)), 1)
    sample_start = max(int(sample_start), 0)
    sample_stride = max(int(sample_stride), 1)
    rank_seed_offset = int(sample_start) * 1000003
    pair_budget = int(max_samples) if max_samples is not None else int(len(schedules) * replay_multiplier)
    if pair_budget <= 0:
        raise ValueError(f"`max_samples` must be positive, got {pair_budget}.")

    schedule_visits_total = int(len(schedules) * replay_multiplier)

    def _sample_index_stream(target_count: int) -> list[int]:
        if target_count <= 0:
            return []
        indices: list[int] = []
        permutation_round = 0
        while len(indices) < target_count:
            generator = torch.Generator(device="cpu").manual_seed(seed + permutation_round * 7919)
            permutation = torch.randperm(len(dataset), generator=generator).tolist()
            indices.extend(permutation[sample_start::sample_stride])
            permutation_round += 1
        return indices[:target_count]

    dataset_indices = _sample_index_stream(pair_budget)
    if tqdm is not None:
        dataset_indices = tqdm(dataset_indices, desc="Sampling schedule replay x_t", unit="sample")

    schedule_start = 0 if len(schedules) == 0 else (rank_seed_offset % len(schedules))
    records: list[dict[str, Any]] = []
    for pair_idx, dataset_index in enumerate(dataset_indices):
        schedule_index = int((schedule_start + pair_idx) % len(schedules))
        records.extend(
            _sample_one(
                model,
                dataset[int(dataset_index)],
                int(dataset_index),
                schedule=schedules[schedule_index],
                schedule_index=schedule_index,
                num_inference_steps=num_steps,
                seed=seed + int(dataset_index) + rank_seed_offset,
                rand_device=rand_device,
                xt_dtype=torch.float16,
            )
        )
        if max_records is not None and len(records) >= int(max_records):
            records = records[: int(max_records)]
            break

    meta = {
        "ckpt": str(weight_ckpt),
        "schedule_path": str(Path(schedule_path).resolve()),
        "num_schedules": int(len(schedules)),
        "replay_multiplier": int(replay_multiplier),
        "schedule_visits_total": int(schedule_visits_total),
        "pair_budget": int(pair_budget),
        "num_inference_steps": num_steps,
        "seed": seed,
        "xt_dtype": "fp16",
        "records": len(records),
    }
    if save_to_disk:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for old_pt in out_dir.glob("*.pt"):
            old_pt.unlink()
        torch.save({"meta": meta, "records": records}, out_dir / "replay.pt")
        with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
    del model, dataset
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return records


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-config", default="libero_streaming_action_ft_2cam224_1e-4")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--schedule-path", required=True)
    parser.add_argument("--output-dir", default="data/trajectory_replay/libero_schedule_xt")
    parser.add_argument("--max-samples", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default=None)
    parser.add_argument("--rand-device", default="cpu")
    parser.add_argument("--max-records", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    with initialize_config_dir(version_base="1.3", config_dir=str((project_root / "configs").resolve())):
        cfg = compose(
            config_name="train",
            overrides=[
                f"task={args.task_config}",
            ],
        )
    records = collect_xt_replay_from_schedule(
        cfg,
        ckpt=str(args.ckpt),
        schedule_path=args.schedule_path,
        output_dir=args.output_dir,
        max_samples=int(args.max_samples),
        seed=int(args.seed),
        device=args.device,
        rand_device=str(args.rand_device),
        max_records=args.max_records,
    )
    print(f"[sample_schedule_xt] saved {len(records)} records to {Path(args.output_dir) / 'replay.pt'}")


if __name__ == "__main__":
    main()
