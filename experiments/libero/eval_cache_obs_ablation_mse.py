from __future__ import annotations

import argparse
import json
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
    _batched,
    _load_schedule_pool,
    _mixed_precision_to_model_dtype,
    _normalize_mixed_precision,
    resolve_weight_checkpoint,
)
from experiments.libero.eval_schedule_denoise_mse import _run_schedule_for_loaded_sample  # noqa: E402
from fastwam.utils.pytorch_utils import set_global_seed  # noqa: E402


def _register_resolver_if_needed(name: str, fn) -> None:
    if not OmegaConf.has_resolver(name):
        OmegaConf.register_new_resolver(name, fn)


_register_resolver_if_needed("eval", eval)
_register_resolver_if_needed("max", lambda x: max(x))
_register_resolver_if_needed("split", lambda s, idx: s.split("/")[int(idx)])


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


OBS_KEY_BY_CACHE_KEY = {
    "prev": "obs_prev",
    "cur": "obs_cur",
    "curr": "obs_cur",
    "current": "obs_cur",
    "next": "obs_next",
    "next2": "obs_next2",
}


def _parse_mask_sets(text: str) -> list[tuple[str, list[str]]]:
    out: list[tuple[str, list[str]]] = []
    for raw_item in str(text).split(","):
        item = raw_item.strip()
        if not item:
            continue
        if item.lower() in {"none", "clean", "baseline"}:
            out.append(("none", []))
            continue
        keys = [part.strip().lower() for part in item.replace("+", "|").split("|") if part.strip()]
        invalid = [key for key in keys if key not in OBS_KEY_BY_CACHE_KEY]
        if invalid:
            raise ValueError(f"Invalid mask key(s): {invalid}. Valid: {sorted(OBS_KEY_BY_CACHE_KEY)}")
        name = "+".join(keys)
        out.append((name, keys))
    if not out:
        raise ValueError("No mask sets selected.")
    return out


def _clone_sample_with_noise(
    sample: dict[str, Any],
    *,
    mask_keys: list[str],
    generator: torch.Generator,
    noise_mode: str,
) -> dict[str, Any]:
    out = dict(sample)
    obs_keys = sorted({OBS_KEY_BY_CACHE_KEY[key] for key in mask_keys})
    for obs_key in obs_keys:
        tensor = out.get(obs_key)
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"Sample field {obs_key!r} is missing or not a tensor.")
        if str(noise_mode) == "gaussian":
            noise = torch.randn(tensor.shape, generator=generator, dtype=tensor.dtype)
        elif str(noise_mode) == "uniform":
            noise = torch.rand(tensor.shape, generator=generator, dtype=tensor.dtype).mul_(2.0).sub_(1.0)
        else:
            raise ValueError(f"Unsupported noise_mode={noise_mode}. Expected gaussian or uniform.")
        out[obs_key] = noise.to(device=tensor.device)
    return out


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / float(len(values)))


def _summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    final_mses = [float(row["final_mse"]) for row in rows]
    final_env_mses = [
        float(row["final_env_action_mse"])
        for row in rows
        if row.get("final_env_action_mse") is not None
    ]
    return {
        "count": int(len(rows)),
        "final_mse_mean": _mean(final_mses),
        "final_mse_min": None if not final_mses else float(min(final_mses)),
        "final_mse_max": None if not final_mses else float(max(final_mses)),
        "final_env_action_mse_mean": _mean(final_env_mses),
        "final_env_action_mse_min": None if not final_env_mses else float(min(final_env_mses)),
        "final_env_action_mse_max": None if not final_env_mses else float(max(final_env_mses)),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Ablate prev/cur/next/next2 cache observations by replacing selected obs with pure noise, "
            "then run full denoise under a sampled async schedule and report MSE against GT action."
        )
    )
    parser.add_argument("--task-config", default="libero_streaming_action_ft_2cam224_1e-4")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--schedule-path", required=True)
    parser.add_argument(
        "--schedule-index",
        type=int,
        default=-1,
        help="Schedule index. Use -1 to sample randomly.",
    )
    parser.add_argument(
        "--dataset-index",
        type=int,
        default=-1,
        help="Dataset sample index. Use -1 to sample randomly.",
    )
    parser.add_argument("--num-cases", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default=None)
    parser.add_argument("--rand-device", default="cpu")
    parser.add_argument("--num-inference-steps", type=int, default=None)
    parser.add_argument(
        "--mask-sets",
        default="none,prev,cur,next,next2",
        help="Comma-separated ablations. Use + or | inside one item for combinations, e.g. none,prev,cur,next,prev+cur.",
    )
    parser.add_argument("--noise-mode", choices=["gaussian", "uniform"], default="gaussian")
    parser.add_argument("--output-json", default=None)
    return parser.parse_args()


@torch.no_grad()
def run_cache_obs_ablation(cfg, args: argparse.Namespace) -> dict[str, Any]:
    set_global_seed(max(int(args.seed), 1))
    schedules = _load_schedule_pool(args.schedule_path)
    if len(schedules) == 0:
        raise ValueError(f"No schedules available: {args.schedule_path}")

    sample_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    with open_dict(sample_cfg.data.train):
        sample_cfg.data.train.pop("trajectory_replay_key", None)
    dataset = instantiate(sample_cfg.data.train)
    if len(dataset) <= 0:
        raise ValueError("Dataset is empty.")

    mixed_precision = _normalize_mixed_precision(str(cfg.mixed_precision))
    model_dtype = _mixed_precision_to_model_dtype(mixed_precision)
    model_device = str(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    model = instantiate(cfg.model, model_dtype=model_dtype, device=model_device)
    weight_ckpt = resolve_weight_checkpoint(str(args.ckpt))
    model.load_checkpoint(str(weight_ckpt))
    model.eval()

    processor = getattr(getattr(dataset, "lerobot_dataset", None), "processor", None)
    binarize_gripper = bool(cfg.get("EVALUATION", {}).get("binarize_gripper", False))
    infer_steps = (
        int(args.num_inference_steps)
        if args.num_inference_steps is not None
        else int(cfg.model.streaming.streaming_train.get("infer_num_inference_steps", 10))
    )
    mask_sets = _parse_mask_sets(str(args.mask_sets))
    selector = torch.Generator(device="cpu").manual_seed(max(int(args.seed), 1) + 41017)

    cases: list[dict[str, Any]] = []
    rows_by_ablation: dict[str, list[dict[str, Any]]] = {name: [] for name, _ in mask_sets}
    for case_idx in range(max(int(args.num_cases), 1)):
        dataset_index = (
            int(torch.randint(low=0, high=len(dataset), size=(1,), generator=selector).item())
            if int(args.dataset_index) < 0
            else int(args.dataset_index)
        )
        schedule_index = (
            int(torch.randint(low=0, high=len(schedules), size=(1,), generator=selector).item())
            if int(args.schedule_index) < 0
            else int(args.schedule_index)
        )
        if not 0 <= dataset_index < len(dataset):
            raise IndexError(f"dataset_index={dataset_index} out of range [0, {len(dataset) - 1}]")
        if not 0 <= schedule_index < len(schedules):
            raise IndexError(f"schedule_index={schedule_index} out of range [0, {len(schedules) - 1}]")

        sample = dataset[dataset_index]
        case_seed = int(args.seed) + int(case_idx) * 100003
        case_rows: list[dict[str, Any]] = []
        baseline_final_mse: float | None = None
        baseline_env_mse: float | None = None
        for ablation_idx, (ablation_name, mask_keys) in enumerate(mask_sets):
            noise_generator = torch.Generator(device="cpu").manual_seed(
                int(case_seed) + int(ablation_idx) * 1009 + int(dataset_index) * 17
            )
            ablated_sample = _clone_sample_with_noise(
                sample,
                mask_keys=mask_keys,
                generator=noise_generator,
                noise_mode=str(args.noise_mode),
            )
            batch = model._extract_streaming_episode_batch(_batched(ablated_sample))
            row = _run_schedule_for_loaded_sample(
                model,
                ablated_sample,
                batch,
                schedule=schedules[schedule_index],
                schedule_index=int(schedule_index),
                seed=int(case_seed) + int(schedule_index),
                rand_device=str(args.rand_device),
                infer_steps=int(infer_steps),
                include_per_step=False,
                processor=processor,
                binarize_gripper=binarize_gripper,
            )
            row.update(
                {
                    "case_index": int(case_idx),
                    "dataset_index": int(dataset_index),
                    "schedule_index": int(schedule_index),
                    "ablation": str(ablation_name),
                    "masked_cache_keys": list(mask_keys),
                    "noise_mode": str(args.noise_mode),
                    "trigger_obs_idx": int(sample.get("trigger_obs_idx", -1)),
                    "raw_action_start": int(sample.get("raw_action_start", -1)),
                }
            )
            if ablation_name == "none":
                baseline_final_mse = float(row["final_mse"])
                baseline_env_mse = (
                    None
                    if row.get("final_env_action_mse") is None
                    else float(row["final_env_action_mse"])
                )
            if baseline_final_mse is not None:
                row["delta_final_mse_vs_none"] = float(row["final_mse"]) - float(baseline_final_mse)
            if baseline_env_mse is not None and row.get("final_env_action_mse") is not None:
                row["delta_final_env_action_mse_vs_none"] = (
                    float(row["final_env_action_mse"]) - float(baseline_env_mse)
                )
            rows_by_ablation[ablation_name].append(row)
            case_rows.append(row)

        baseline = next((row for row in case_rows if row["ablation"] == "none"), None)
        cases.append(
            {
                "case_index": int(case_idx),
                "dataset_index": int(dataset_index),
                "schedule_index": int(schedule_index),
                "episode_index": int(sample.get("episode_idx", -1)),
                "trigger_obs_idx": int(sample.get("trigger_obs_idx", -1)),
                "raw_action_start": int(sample.get("raw_action_start", -1)),
                "baseline_final_mse": None if baseline is None else float(baseline["final_mse"]),
                "baseline_final_env_action_mse": (
                    None
                    if baseline is None or baseline.get("final_env_action_mse") is None
                    else float(baseline["final_env_action_mse"])
                ),
                "rows": case_rows,
            }
        )

    ablation_summaries: dict[str, Any] = {}
    for name, rows in rows_by_ablation.items():
        summary = _summarize_rows(rows)
        deltas = [
            float(row["delta_final_mse_vs_none"])
            for row in rows
            if row.get("delta_final_mse_vs_none") is not None
        ]
        env_deltas = [
            float(row["delta_final_env_action_mse_vs_none"])
            for row in rows
            if row.get("delta_final_env_action_mse_vs_none") is not None
        ]
        summary.update(
            {
                "delta_final_mse_vs_none_mean": _mean(deltas),
                "delta_final_env_action_mse_vs_none_mean": _mean(env_deltas),
            }
        )
        ablation_summaries[name] = summary

    result = {
        "ckpt": str(weight_ckpt),
        "task_config": str(args.task_config),
        "schedule_path": str(Path(args.schedule_path).resolve()),
        "seed": int(args.seed),
        "device": str(model_device),
        "rand_device": str(args.rand_device),
        "noise_mode": str(args.noise_mode),
        "num_cases": int(len(cases)),
        "num_dataset_samples_total": int(len(dataset)),
        "num_schedules_total": int(len(schedules)),
        "num_inference_steps": int(infer_steps),
        "mask_sets": [{"name": name, "keys": keys} for name, keys in mask_sets],
        "summary_by_ablation": ablation_summaries,
        "cases": cases,
    }
    del model, dataset
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def main() -> None:
    args = _parse_args()
    with initialize_config_dir(version_base="1.3", config_dir=str((project_root / "configs").resolve())):
        cfg = compose(config_name="train", overrides=[f"task={args.task_config}"])
    result = run_cache_obs_ablation(cfg, args)
    text = json.dumps(result, indent=2, cls=NumpyEncoder)
    print(text)
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")
        print(f"Wrote cache obs ablation MSE to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
