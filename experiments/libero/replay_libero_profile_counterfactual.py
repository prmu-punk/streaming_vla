from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pyarrow.parquet as pq
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from experiments.libero.compare_libero_native_async_utils import collect_episode_obs_trace  # noqa: E402
from experiments.libero.eval_libero_policy_utils import _obs_to_model_input, _postprocess_libero_action_chunk  # noqa: E402
from experiments.libero.eval_libero_single import (  # noqa: E402
    NumpyEncoder,
    _build_eval_model,
    _configure_egl_device,
    _mixed_precision_to_model_dtype,
    _resolve_dataset_stats_path,
    _resolve_eval_device,
)
from experiments.libero.libero_utils import invert_gripper_action  # noqa: E402
from fastwam.datasets.lerobot.processors.fastwam_processor import FastWAMProcessor  # noqa: E402
from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT  # noqa: E402
from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json  # noqa: E402
from fastwam.models.wan22.streaming_cache import CacheSnapshot  # noqa: E402
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


@dataclass(frozen=True)
class ProfileStepTrace:
    denoise_step: int
    layer_obs_indices: list[int]
    mode: str
    frontier: int
    latest_offset: int
    older_offset: Optional[int]


@dataclass(frozen=True)
class ProfileJobTrace:
    episode_idx: int
    job_id: int
    trigger_env_step: int
    trigger_obs_index: int
    steps: list[ProfileStepTrace]


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    perturb_sources: list[str]
    perturb_steps: Optional[list[int]]
    selected_steps_use_full_wrong_cache: bool = False


@dataclass
class CacheBankEntry:
    obs_index: int
    obs_timestamp_ms: float
    video_seq_len: int
    tokens_per_frame: int
    cache_layers: list[dict[str, torch.Tensor]]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay profiled LIBERO async jobs offline under counterfactual cache corruption. "
            "The environment trajectory is reconstructed via GT replay once per episode, then each "
            "job is replayed offline using the profiled per-step layer source trace."
        )
    )
    parser.add_argument("--profile-json", required=True)
    parser.add_argument("--streaming-task-config", required=True)
    parser.add_argument("--streaming-ckpt", required=True)
    parser.add_argument("--task-suite-name", required=True)
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--mixed-precision", default="bf16")
    parser.add_argument("--device", default=None)
    parser.add_argument("--cache-device", default=None)
    parser.add_argument("--action-device", default=None)
    parser.add_argument("--render-gpu-id", type=int, default=0)
    parser.add_argument("--num-inference-steps", type=int, default=None)
    parser.add_argument("--action-horizon", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--gt-episode-index", type=int, default=None)
    parser.add_argument("--episode-idx", type=int, default=None)
    parser.add_argument("--max-jobs-per-episode", type=int, default=None)
    parser.add_argument("--source-for-step-tests", default="curr")
    parser.add_argument("--wrong-obs-seed", type=int, default=0)
    parser.add_argument("--save-job-results", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    return parser.parse_args()


def _compose_cfg(task_config_name: str, ckpt: str, args: argparse.Namespace) -> DictConfig:
    overrides = [
        f"task={task_config_name}",
        f"ckpt={ckpt}",
        f"EVALUATION.task_suite_name={args.task_suite_name}",
        f"EVALUATION.task_id={int(args.task_id)}",
        "EVALUATION.num_trials=1",
        f"gpu_id={int(args.render_gpu_id)}",
        f"mixed_precision={args.mixed_precision}",
    ]
    if args.device is not None:
        overrides.append(f"EVALUATION.device={args.device}")
    if args.num_inference_steps is not None:
        overrides.append(f"EVALUATION.num_inference_steps={int(args.num_inference_steps)}")
    if args.action_horizon is not None:
        overrides.append(f"EVALUATION.action_horizon={int(args.action_horizon)}")
    if args.seed is not None:
        overrides.append(f"seed={int(args.seed)}")

    config_dir = str((project_root / "configs").resolve())
    with initialize_config_dir(version_base="1.3", config_dir=config_dir):
        return compose(config_name="sim_libero", overrides=overrides)


def _build_processor(cfg: DictConfig) -> tuple[FastWAMProcessor, Path]:
    dataset_stats_path = _resolve_dataset_stats_path(cfg)
    dataset_stats = load_dataset_stats_from_json(str(dataset_stats_path))
    processor: FastWAMProcessor = instantiate(cfg.data.train.processor).eval()
    processor.set_normalizer_from_stats(dataset_stats)
    return processor, dataset_stats_path


def _normalize_runtime_device(device: str, *, render_gpu_id: int) -> str:
    resolved = str(device)
    if resolved == "cuda":
        return f"cuda:{int(render_gpu_id)}"
    return resolved


def _resolve_action_horizon(cfg: DictConfig) -> int:
    horizon_cfg = cfg.EVALUATION.get("action_horizon", None)
    action_horizon = int(cfg.data.train.num_frames) - 1 if horizon_cfg is None else int(horizon_cfg)
    if action_horizon <= 0:
        raise ValueError(f"Resolved action horizon must be positive, got {action_horizon}")
    return action_horizon


def _resolve_input_hw(cfg: DictConfig) -> tuple[int, int]:
    video_size = cfg.data.train.get("video_size", [224, 224])
    return int(video_size[1]), int(video_size[0])


def _resolve_task_dataset_root(cfg: DictConfig, task_suite_name: str) -> Path:
    dataset_dirs = [Path(str(v)).resolve() for v in cfg.data.train.get("dataset_dirs", [])]
    for dataset_dir in dataset_dirs:
        if task_suite_name in dataset_dir.name:
            return dataset_dir
    raise FileNotFoundError(
        f"Failed to find dataset dir for suite={task_suite_name}. Candidates: {[str(v) for v in dataset_dirs]}"
    )


def _find_task_episode_indices(dataset_root: Path, task_description: str) -> list[int]:
    episodes_path = dataset_root / "meta" / "episodes.jsonl"
    matched: list[int] = []
    with open(episodes_path, "r", encoding="utf-8") as f:
        for line in f:
            payload = json.loads(line)
            tasks = list(payload.get("tasks", []))
            if len(tasks) > 0 and str(tasks[0]) == str(task_description):
                matched.append(int(payload["episode_index"]))
    if not matched:
        raise ValueError(f"No GT episodes found for task description: {task_description}")
    return matched


def _read_episode_actions(dataset_root: Path, episode_index: int) -> np.ndarray:
    parquet_path = dataset_root / "data" / f"chunk-{episode_index // 1000:03d}" / f"episode_{episode_index:06d}.parquet"
    table = pq.read_table(parquet_path, columns=["action"])
    actions = np.asarray(table["action"].to_pylist(), dtype=np.float32)
    if actions.ndim != 2:
        raise ValueError(f"Expected [T, D] actions, got {actions.shape}")
    return actions


def _dataset_action_to_env_action(action: np.ndarray, *, binarize_gripper: bool) -> np.ndarray:
    converted = np.asarray(action, dtype=np.float32).copy()
    converted[..., -1] = converted[..., -1] * 2.0 - 1.0
    converted = invert_gripper_action(converted)
    if binarize_gripper:
        converted[..., -1] = np.sign(converted[..., -1])
    return converted


def clone_obs_dict(obs_dict: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(obs_dict)


def classify_source(*, layer_obs_index: int, trigger_obs_index: int) -> str:
    offset = int(layer_obs_index) - int(trigger_obs_index)
    if offset < 0:
        return "prev"
    if offset == 0:
        return "curr"
    return "future"


def is_valid_profile_job(job_payload: dict) -> bool:
    trigger_obs_index = int(job_payload.get("trigger_obs_index", -1))
    if trigger_obs_index < 0 and int(job_payload.get("trigger_env_step", -1)) == 0:
        return False
    steps = list(job_payload.get("job_layer_source_steps", []))
    if len(steps) == 0:
        return False
    for step in steps:
        latest_offset = int(step.get("latest_offset", 0))
        if abs(latest_offset) > 5:
            return False
    return True


def load_profile_jobs(
    *,
    profile_json: Path,
    episode_idx: int | None = None,
    max_jobs_per_episode: int | None = None,
) -> list[ProfileJobTrace]:
    with open(profile_json, "r", encoding="utf-8") as f:
        payload = json.load(f)

    episodes = list(payload.get("async_runtime_episodes", []))
    jobs: list[ProfileJobTrace] = []
    for ep in episodes:
        resolved_episode_idx = int(ep.get("episode_idx", -1))
        if episode_idx is not None and resolved_episode_idx != int(episode_idx):
            continue
        kept = 0
        for job in list(ep.get("job_records", [])):
            if not is_valid_profile_job(job):
                continue
            if max_jobs_per_episode is not None and kept >= int(max_jobs_per_episode):
                break
            step_rows: list[ProfileStepTrace] = []
            for step in list(job.get("job_layer_source_steps", [])):
                step_rows.append(
                    ProfileStepTrace(
                        denoise_step=int(step["denoise_step"]),
                        layer_obs_indices=[int(v) for v in list(step["layer_obs_indices"])],
                        mode=str(step["mode"]),
                        frontier=int(step["frontier"]),
                        latest_offset=int(step["latest_offset"]),
                        older_offset=(
                            None if step.get("older_offset") is None else int(step.get("older_offset"))
                        ),
                    )
                )
            jobs.append(
                ProfileJobTrace(
                    episode_idx=resolved_episode_idx,
                    job_id=int(job["job_id"]),
                    trigger_env_step=int(job["trigger_env_step"]),
                    trigger_obs_index=int(job["trigger_obs_index"]),
                    steps=step_rows,
                )
            )
            kept += 1
    return jobs


def build_experiment_specs(*, source_for_step_tests: str) -> list[ExperimentSpec]:
    _ = str(source_for_step_tests)  # Kept for CLI compatibility; step tests now ablate the full cache.
    return [
        ExperimentSpec(name="clean", perturb_sources=[], perturb_steps=None),
        ExperimentSpec(name="all-prev-wrong", perturb_sources=["prev"], perturb_steps=None),
        ExperimentSpec(name="all-curr-wrong", perturb_sources=["curr"], perturb_steps=None),
        ExperimentSpec(name="all-future-wrong", perturb_sources=["future"], perturb_steps=None),
        ExperimentSpec(
            name="steps06-fullcache-wrong",
            perturb_sources=[],
            perturb_steps=[0, 5],
            selected_steps_use_full_wrong_cache=True,
        ),
        ExperimentSpec(
            name="steps6plus-fullcache-wrong",
            perturb_sources=[],
            perturb_steps=list(range(6, 64)),
            selected_steps_use_full_wrong_cache=True,
        ),
    ]


def summarize_job_sources(job: ProfileJobTrace) -> dict:
    step_rows: list[dict] = []
    for step in job.steps:
        source_counts = {"prev": 0, "curr": 0, "future": 0}
        for obs_index in step.layer_obs_indices:
            source = classify_source(layer_obs_index=obs_index, trigger_obs_index=job.trigger_obs_index)
            source_counts[source] += 1
        step_rows.append(
            {
                "denoise_step": int(step.denoise_step),
                "mode": str(step.mode),
                "frontier": int(step.frontier),
                "latest_offset": int(step.latest_offset),
                "older_offset": (None if step.older_offset is None else int(step.older_offset)),
                "source_counts": source_counts,
            }
        )
    return {
        "episode_idx": int(job.episode_idx),
        "job_id": int(job.job_id),
        "trigger_env_step": int(job.trigger_env_step),
        "trigger_obs_index": int(job.trigger_obs_index),
        "steps": step_rows,
    }


def build_counterfactual_plan(*, jobs: list[ProfileJobTrace], experiments: list[ExperimentSpec]) -> dict:
    return {
        "num_jobs": int(len(jobs)),
        "num_experiments": int(len(experiments)),
        "experiments": [asdict(exp) for exp in experiments],
        "jobs": [summarize_job_sources(job) for job in jobs],
    }


def _build_profile_obs_index_schedule(
    *,
    obs_trace: list[dict[str, Any]],
    obs_stride_env_steps: int,
    control_dt_ms: float,
    jobs: list[ProfileJobTrace],
    required_obs_indices: set[int],
) -> dict[int, tuple[dict[str, Any], float]]:
    if len(obs_trace) == 0:
        raise ValueError("obs_trace must be non-empty.")
    if len(jobs) == 0:
        raise ValueError("jobs must be non-empty.")
    if len(required_obs_indices) == 0:
        return {}

    schedule: dict[int, tuple[dict[str, Any], float]] = {}

    formal_start_candidates = {
        int(job.trigger_obs_index) - (int(job.trigger_env_step) // int(obs_stride_env_steps))
        for job in jobs
        if int(job.trigger_env_step) % int(obs_stride_env_steps) == 0
    }
    if len(formal_start_candidates) != 1:
        raise ValueError(
            f"Failed to infer a unique formal obs_index start from profile jobs: {sorted(formal_start_candidates)}"
        )
    formal_obs_index_start = int(next(iter(formal_start_candidates)))

    min_required_obs_index = int(min(required_obs_indices))
    max_required_obs_index = int(max(required_obs_indices))

    for obs_index in range(min_required_obs_index, max_required_obs_index + 1):
        if obs_index < formal_obs_index_start:
            obs_dict = clone_obs_dict(obs_trace[0])
        else:
            formal_obs_offset = int(obs_index) - int(formal_obs_index_start)
            env_step = int(formal_obs_offset) * int(obs_stride_env_steps)
            if env_step < 0:
                raise IndexError(
                    f"Profile requires obs_index={obs_index} -> env_step={env_step}, "
                    f"but obs_trace only has {len(obs_trace)} steps."
                )
            if env_step >= len(obs_trace):
                obs_dict = clone_obs_dict(obs_trace[-1])
            else:
                obs_dict = clone_obs_dict(obs_trace[env_step])
        schedule[int(obs_index)] = (
            obs_dict,
            float(obs_index) * float(obs_stride_env_steps) * float(control_dt_ms),
        )
    return schedule


def _encode_obs_cpu(
    obs_dict: dict[str, Any],
    *,
    cfg: DictConfig,
    processor: FastWAMProcessor,
    width: int,
    height: int,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    return _obs_to_model_input(
        obs_dict,
        cfg=cfg,
        processor=processor,
        width=width,
        height=height,
        device="cpu",
        dtype=torch.float32,
    )


def _make_wrong_obs(
    obs_dict: dict[str, Any],
    *,
    seed: int,
) -> dict[str, Any]:
    # NumPy requires a non-negative seed, while profiled warmup obs indices can be negative.
    rng = np.random.default_rng(int(seed) & ((1 << 64) - 1))
    wrong_obs = clone_obs_dict(obs_dict)
    for image_key in ("agentview_image", "robot0_eye_in_hand_image"):
        if image_key not in wrong_obs:
            continue
        image = np.asarray(wrong_obs[image_key])
        noise = rng.integers(0, 256, size=image.shape, dtype=np.uint8)
        if image.dtype != np.uint8:
            noise = noise.astype(image.dtype, copy=False)
        wrong_obs[image_key] = noise
    return wrong_obs


def _clone_cache_layers_to_cpu(cache_layers: list[dict[str, Any]]) -> list[dict[str, torch.Tensor]]:
    rows: list[dict[str, torch.Tensor]] = []
    for layer in cache_layers:
        rows.append(
            {
                "k": layer["k"].detach().to(device="cpu"),
                "v": layer["v"].detach().to(device="cpu"),
                "source_delta": int(layer.get("source_delta", 0)),
            }
        )
    return rows


def _build_single_cache_bank_entry(
    *,
    video_model: torch.nn.Module,
    image_cpu: torch.Tensor,
    context: torch.Tensor,
    context_mask: torch.Tensor,
    obs_index: int,
    obs_timestamp_ms: float,
    tiled: bool,
) -> CacheBankEntry:
    version = video_model._build_cache_version(  # type: ignore[attr-defined]
        input_image=image_cpu.to(device=video_model.device, dtype=video_model.torch_dtype),
        context=context,
        context_mask=context_mask,
        obs_index=int(obs_index),
        obs_timestamp_ms=float(obs_timestamp_ms),
        tiled=bool(tiled),
    )
    return CacheBankEntry(
        obs_index=int(obs_index),
        obs_timestamp_ms=float(obs_timestamp_ms),
        video_seq_len=int(version.video_seq_len),
        tokens_per_frame=int(version.tokens_per_frame),
        cache_layers=_clone_cache_layers_to_cpu(version.cache_layers),  # type: ignore[arg-type]
    )


def build_cache_banks_for_episode(
    *,
    obs_schedule: dict[int, tuple[dict[str, Any], float]],
    obs_indices_for_clean: set[int],
    obs_indices_for_wrong: set[int],
    video_model: torch.nn.Module,
    context: torch.Tensor,
    context_mask: torch.Tensor,
    cfg: DictConfig,
    processor: FastWAMProcessor,
    width: int,
    height: int,
    wrong_obs_seed: int,
) -> tuple[dict[int, CacheBankEntry], dict[int, CacheBankEntry]]:
    clean_bank: dict[int, CacheBankEntry] = {}
    wrong_bank: dict[int, CacheBankEntry] = {}
    tiled = bool(cfg.EVALUATION.get("tiled", False))

    for obs_index in sorted(obs_indices_for_clean):
        obs_dict, obs_timestamp_ms = obs_schedule[int(obs_index)]
        image_cpu, _, _ = _encode_obs_cpu(
            obs_dict,
            cfg=cfg,
            processor=processor,
            width=width,
            height=height,
        )
        clean_bank[int(obs_index)] = _build_single_cache_bank_entry(
            video_model=video_model,
            image_cpu=image_cpu,
            context=context,
            context_mask=context_mask,
            obs_index=int(obs_index),
            obs_timestamp_ms=float(obs_timestamp_ms),
            tiled=tiled,
        )

    for obs_index in sorted(obs_indices_for_wrong):
        obs_dict, obs_timestamp_ms = obs_schedule[int(obs_index)]
        wrong_obs = _make_wrong_obs(
            obs_dict,
            seed=int(wrong_obs_seed) + int(obs_index),
        )
        image_cpu, _, _ = _encode_obs_cpu(
            wrong_obs,
            cfg=cfg,
            processor=processor,
            width=width,
            height=height,
        )
        wrong_bank[int(obs_index)] = _build_single_cache_bank_entry(
            video_model=video_model,
            image_cpu=image_cpu,
            context=context,
            context_mask=context_mask,
            obs_index=int(obs_index),
            obs_timestamp_ms=float(obs_timestamp_ms),
            tiled=tiled,
        )
    return clean_bank, wrong_bank


def _compute_frontier(layer_obs_indices: list[int]) -> int:
    if len(layer_obs_indices) == 0:
        return 0
    latest_obs_index = max(int(v) for v in layer_obs_indices)
    frontier = 0
    while frontier < len(layer_obs_indices) and int(layer_obs_indices[frontier]) == latest_obs_index:
        frontier += 1
    return int(len(layer_obs_indices) if frontier == 0 else frontier)


def _get_required_wrong_obs_indices(*, jobs: list[ProfileJobTrace], experiment: ExperimentSpec) -> set[int]:
    required: set[int] = set()
    step_filter = None if experiment.perturb_steps is None else set(int(v) for v in experiment.perturb_steps)
    source_filter = set(str(v) for v in experiment.perturb_sources)
    if not experiment.selected_steps_use_full_wrong_cache and len(source_filter) == 0:
        return required
    for job in jobs:
        for step in job.steps:
            if step_filter is not None and int(step.denoise_step) not in step_filter:
                continue
            if experiment.selected_steps_use_full_wrong_cache:
                required.update(int(v) for v in step.layer_obs_indices)
                continue
            for obs_index in step.layer_obs_indices:
                source = classify_source(layer_obs_index=obs_index, trigger_obs_index=job.trigger_obs_index)
                if source in source_filter:
                    required.add(int(obs_index))
    return required


def _get_required_wrong_obs_indices_for_job(*, job: ProfileJobTrace, experiment: ExperimentSpec) -> set[int]:
    required: set[int] = set()
    step_filter = None if experiment.perturb_steps is None else set(int(v) for v in experiment.perturb_steps)
    source_filter = set(str(v) for v in experiment.perturb_sources)
    if not experiment.selected_steps_use_full_wrong_cache and len(source_filter) == 0:
        return required
    for step in job.steps:
        if step_filter is not None and int(step.denoise_step) not in step_filter:
            continue
        if experiment.selected_steps_use_full_wrong_cache:
            required.update(int(v) for v in step.layer_obs_indices)
            continue
        for obs_index in step.layer_obs_indices:
            source = classify_source(layer_obs_index=int(obs_index), trigger_obs_index=int(job.trigger_obs_index))
            if source in source_filter:
                required.add(int(obs_index))
    return required


def _infer_formal_obs_index_start(*, jobs: list[ProfileJobTrace], obs_stride_env_steps: int) -> int:
    candidates = {
        int(job.trigger_obs_index) - (int(job.trigger_env_step) // int(obs_stride_env_steps))
        for job in jobs
        if int(job.trigger_env_step) % int(obs_stride_env_steps) == 0
    }
    if len(candidates) != 1:
        raise ValueError(f"Failed to infer unique formal obs start: {sorted(candidates)}")
    return int(next(iter(candidates)))


def _move_layer_to_device(layer_cache: dict[str, torch.Tensor], *, device: str, dtype: torch.dtype) -> dict[str, Any]:
    return {
        "k": layer_cache["k"].to(device=device, non_blocking=True),
        "v": layer_cache["v"].to(device=device, non_blocking=True),
        "source_delta": int(layer_cache.get("source_delta", 0)),
    }


def _build_counterfactual_snapshot(
    *,
    step_trace: ProfileStepTrace,
    trigger_obs_index: int,
    experiment: ExperimentSpec,
    clean_bank: dict[int, CacheBankEntry],
    wrong_bank: dict[int, CacheBankEntry],
    action_context: torch.Tensor,
    action_context_mask: torch.Tensor,
    action_device: str,
    action_dtype: torch.dtype,
) -> CacheSnapshot:
    step_filter = None if experiment.perturb_steps is None else set(int(v) for v in experiment.perturb_steps)
    source_filter = set(str(v) for v in experiment.perturb_sources)
    use_step_perturb = step_filter is None or int(step_trace.denoise_step) in step_filter
    use_full_wrong_cache = bool(experiment.selected_steps_use_full_wrong_cache and use_step_perturb)

    layer_rows: list[dict[str, Any]] = []
    layer_obs_indices = [int(v) for v in step_trace.layer_obs_indices]
    reference_entry = clean_bank[int(layer_obs_indices[0])]
    for layer_idx, obs_index in enumerate(layer_obs_indices):
        source = classify_source(layer_obs_index=int(obs_index), trigger_obs_index=int(trigger_obs_index))
        use_wrong = bool(use_full_wrong_cache or (use_step_perturb and source in source_filter))
        bank = wrong_bank if use_wrong else clean_bank
        entry = bank[int(obs_index)]
        layer_rows.append(
            _move_layer_to_device(
                entry.cache_layers[layer_idx],
                device=action_device,
                dtype=action_dtype,
            )
        )

    frontier = _compute_frontier(layer_obs_indices)
    latest_obs_index = max(layer_obs_indices)
    return CacheSnapshot(
        version=int(step_trace.denoise_step),
        obs_index=int(latest_obs_index),
        obs_timestamp_ms=float(reference_entry.obs_timestamp_ms),
        frontier=int(frontier),
        video_seq_len=int(reference_entry.video_seq_len),
        tokens_per_frame=int(reference_entry.tokens_per_frame),
        cache_layers=layer_rows,
        context=action_context,
        context_mask=action_context_mask,
        layer_version_ids=[int(step_trace.denoise_step)] * len(layer_obs_indices),
        layer_obs_indices=layer_obs_indices,
        layer_obs_timestamps_ms=[float(reference_entry.obs_timestamp_ms)] * len(layer_obs_indices),
        layer_ready_events=[None] * len(layer_obs_indices),
    )


def replay_one_job_counterfactual(
    *,
    job: ProfileJobTrace,
    experiment: ExperimentSpec,
    action_model: torch.nn.Module,
    action_context: torch.Tensor,
    action_context_mask: torch.Tensor,
    trigger_proprio: torch.Tensor,
    clean_bank: dict[int, CacheBankEntry],
    wrong_bank: dict[int, CacheBankEntry],
    action_horizon: int,
    num_inference_steps: int,
    sigma_shift: Optional[float],
    rand_device: str,
    seed: Optional[int],
    processor: FastWAMProcessor,
    cfg: DictConfig,
) -> np.ndarray:
    action_model.reset_streaming_state()
    if len(job.steps) <= 0:
        raise ValueError(f"Job {job.job_id} has no profiled denoise steps.")
    with torch.no_grad():
        job_state = action_model.start_action_job(
            action_horizon=int(action_horizon),
            context=action_context,
            context_mask=action_context_mask,
            proprio=trigger_proprio.to(device=action_model.device, dtype=action_model.torch_dtype),
            trigger_obs_index=int(job.trigger_obs_index),
            num_inference_steps=int(num_inference_steps),
            sigma_shift=sigma_shift,
            seed=seed,
            rand_device=str(rand_device),
        )
        while not job_state.done:
            step_idx = int(job_state.current_step_idx)
            step_trace = job.steps[step_idx]
            snapshot = _build_counterfactual_snapshot(
                step_trace=step_trace,
                trigger_obs_index=int(job.trigger_obs_index),
                experiment=experiment,
                clean_bank=clean_bank,
                wrong_bank=wrong_bank,
                action_context=action_context,
                action_context_mask=action_context_mask,
                action_device=str(action_model.device),
                action_dtype=action_model.torch_dtype,
            )
            action_model.step_action_job(job_state, snapshot=snapshot)
    return _postprocess_libero_action_chunk(job_state.latents_action, processor=processor, cfg=cfg)


def _compute_chunk_metrics(*, pred_chunk: np.ndarray, gt_actions: np.ndarray, trigger_env_step: int) -> dict[str, Any]:
    pred = np.asarray(pred_chunk, dtype=np.float32)
    gt = np.asarray(gt_actions[trigger_env_step : trigger_env_step + pred.shape[0]], dtype=np.float32)
    valid_len = int(min(pred.shape[0], gt.shape[0]))
    pred = pred[:valid_len]
    gt = gt[:valid_len]
    diff = pred - gt
    head_len = int(min(8, valid_len))
    head_diff = diff[:head_len]
    first_diff = diff[0] if valid_len > 0 else None
    return {
        "chunk_len": int(pred.shape[0]),
        "whole_mse": (None if valid_len == 0 else float(np.mean(np.square(diff)))),
        "whole_mae": (None if valid_len == 0 else float(np.mean(np.abs(diff)))),
        "head_mse": (None if head_len == 0 else float(np.mean(np.square(head_diff)))),
        "head_mae": (None if head_len == 0 else float(np.mean(np.abs(head_diff)))),
        "first_action_mse": (None if first_diff is None else float(np.mean(np.square(first_diff)))),
    }


def _summarize_result_rows(rows: list[dict[str, Any]], *, experiments: list[ExperimentSpec]) -> list[dict[str, Any]]:
    metric_names = [
        "chunk_len",
        "whole_mse",
        "whole_mae",
        "head_mse",
        "head_mae",
        "first_action_mse",
    ]
    summaries: list[dict[str, Any]] = []
    experiment_names = [str(exp.name) for exp in experiments]
    for experiment_name in experiment_names:
        matched = [row for row in rows if str(row.get("experiment")) == experiment_name]
        summary: dict[str, Any] = {
            "experiment": str(experiment_name),
            "num_rows": int(len(matched)),
        }
        for metric_name in metric_names:
            values = [float(row[metric_name]) for row in matched if row.get(metric_name) is not None]
            if len(values) == 0:
                summary[metric_name] = None
                continue
            arr = np.asarray(values, dtype=np.float64)
            summary[metric_name] = {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
            }
        summaries.append(summary)
    return summaries


def _group_jobs_by_episode(jobs: list[ProfileJobTrace]) -> dict[int, list[ProfileJobTrace]]:
    grouped: dict[int, list[ProfileJobTrace]] = {}
    for job in jobs:
        grouped.setdefault(int(job.episode_idx), []).append(job)
    for rows in grouped.values():
        rows.sort(key=lambda item: (int(item.trigger_env_step), int(item.job_id)))
    return grouped


def main() -> None:
    args = _parse_args()
    if args.seed is not None:
        set_global_seed(int(args.seed), get_worker_init_fn=False)

    cfg = _compose_cfg(args.streaming_task_config, args.streaming_ckpt, args)
    _configure_egl_device(cfg)
    model_dtype = _mixed_precision_to_model_dtype(args.mixed_precision)
    base_device = _resolve_eval_device(cfg) if args.device is None else str(args.device)
    base_device = _normalize_runtime_device(base_device, render_gpu_id=int(args.render_gpu_id))
    cache_device = (
        base_device
        if args.cache_device is None
        else _normalize_runtime_device(str(args.cache_device), render_gpu_id=int(args.render_gpu_id))
    )
    action_device = (
        base_device
        if args.action_device is None
        else _normalize_runtime_device(str(args.action_device), render_gpu_id=int(args.render_gpu_id))
    )
    processor, dataset_stats_path = _build_processor(cfg)
    action_horizon = _resolve_action_horizon(cfg)
    input_w, input_h = _resolve_input_hw(cfg)

    profile_jobs = load_profile_jobs(
        profile_json=Path(args.profile_json).resolve(),
        episode_idx=args.episode_idx,
        max_jobs_per_episode=args.max_jobs_per_episode,
    )
    experiments = build_experiment_specs(source_for_step_tests=str(args.source_for_step_tests))
    plan = build_counterfactual_plan(jobs=profile_jobs, experiments=experiments)
    if bool(args.plan_only):
        with open(Path(args.output_json).resolve(), "w", encoding="utf-8") as f:
            json.dump(plan, f, indent=2)
        print(f"Wrote counterfactual replay plan to: {Path(args.output_json).resolve()}")
        return

    if len(profile_jobs) == 0:
        raise ValueError("No valid profile jobs were loaded.")

    cache_model = _build_eval_model(cfg, model_dtype=model_dtype, device=cache_device)
    action_model = cache_model if action_device == cache_device else _build_eval_model(cfg, model_dtype=model_dtype, device=action_device)
    if not hasattr(cache_model, "_build_cache_version"):
        raise ValueError("Cache replay requires a FastWAMStreaming-style model with _build_cache_version().")
    if not hasattr(action_model, "start_action_job"):
        raise ValueError("Action replay requires a FastWAMStreaming-style model with start_action_job().")

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[str(args.task_suite_name)]()
    task = task_suite.get_task(int(args.task_id))
    initial_states = list(task_suite.get_task_init_states(int(args.task_id)))
    dataset_root = _resolve_task_dataset_root(cfg, str(args.task_suite_name))

    prompt = DEFAULT_PROMPT.format(task=str(task.language))
    with torch.no_grad():
        cache_context, cache_context_mask = cache_model.encode_prompt(prompt)
        action_context, action_context_mask = action_model.encode_prompt(prompt)
    cache_context = cache_context.to(device=cache_model.device, dtype=cache_model.torch_dtype)
    cache_context_mask = cache_context_mask.to(device=cache_model.device, dtype=torch.bool)
    action_context = action_context.to(device=action_model.device, dtype=action_model.torch_dtype)
    action_context_mask = action_context_mask.to(device=action_model.device, dtype=torch.bool)

    grouped_jobs = _group_jobs_by_episode(profile_jobs)
    obs_stride_env_steps = int(cfg.EVALUATION.get("async_obs_stride_env_steps", 3))
    trigger_every_n_obs = int(cfg.EVALUATION.get("async_action_trigger_every_n_obs", 3))
    warmup_action_jobs = int(cfg.EVALUATION.get("async_warmup_action_jobs", 0))
    force_first_job = bool(cfg.EVALUATION.get("async_force_first_job", True))
    control_dt_ms = float(cfg.EVALUATION.get("async_control_dt_ms", 50.0))
    sigma_shift = None if cfg.EVALUATION.get("sigma_shift") is None else float(cfg.EVALUATION.get("sigma_shift"))
    num_inference_steps = int(cfg.EVALUATION.get("num_inference_steps", cfg.get("eval_num_inference_steps", 10)))
    rand_device = str(cfg.EVALUATION.get("rand_device", "cpu"))

    results: dict[str, Any] = {
        "profile_json": str(Path(args.profile_json).resolve()),
        "streaming_ckpt": str(args.streaming_ckpt),
        "streaming_task_config": str(args.streaming_task_config),
        "task_suite_name": str(args.task_suite_name),
        "task_id": int(args.task_id),
        "dataset_stats_path": str(dataset_stats_path),
        "cache_device": str(cache_device),
        "action_device": str(action_device),
        "save_job_results": bool(args.save_job_results),
        "experiments": [asdict(exp) for exp in experiments],
        "episodes": [],
    }
    all_result_rows: list[dict[str, Any]] = []

    for episode_idx, episode_jobs in sorted(grouped_jobs.items(), key=lambda kv: kv[0]):
        resolved_trial_idx = int(episode_idx) % len(initial_states)
        initial_state = initial_states[resolved_trial_idx]
        matched_episode_indices = _find_task_episode_indices(dataset_root, str(task.language))
        gt_episode_index = (
            int(args.gt_episode_index)
            if args.gt_episode_index is not None
            else int(matched_episode_indices[resolved_trial_idx % len(matched_episode_indices)])
        )
        gt_actions_raw = _read_episode_actions(dataset_root, gt_episode_index)
        gt_actions_env = _dataset_action_to_env_action(
            gt_actions_raw,
            binarize_gripper=bool(cfg.EVALUATION.get("binarize_gripper", False)),
        )
        obs_trace, task_description = collect_episode_obs_trace(
            task=task,
            initial_state=initial_state,
            cfg=cfg,
            gt_actions_env=gt_actions_env,
            render_gpu_device_id=int(args.render_gpu_id),
        )
        obs_indices_for_clean: set[int] = set()
        for job in episode_jobs:
            for step in job.steps:
                obs_indices_for_clean.update(int(v) for v in step.layer_obs_indices)
        formal_obs_index_start = _infer_formal_obs_index_start(
            jobs=episode_jobs,
            obs_stride_env_steps=obs_stride_env_steps,
        )

        obs_schedule = _build_profile_obs_index_schedule(
            obs_trace=obs_trace,
            obs_stride_env_steps=obs_stride_env_steps,
            control_dt_ms=control_dt_ms,
            jobs=episode_jobs,
            required_obs_indices=obs_indices_for_clean,
        )

        episode_results: list[dict[str, Any]] = []
        for experiment in experiments:
            obs_indices_for_wrong = _get_required_wrong_obs_indices(jobs=episode_jobs, experiment=experiment)
            clean_bank, wrong_bank = build_cache_banks_for_episode(
                obs_schedule=obs_schedule,
                obs_indices_for_clean=obs_indices_for_clean,
                obs_indices_for_wrong=obs_indices_for_wrong,
                video_model=cache_model,
                context=cache_context,
                context_mask=cache_context_mask,
                cfg=cfg,
                processor=processor,
                width=input_w,
                height=input_h,
                wrong_obs_seed=int(args.wrong_obs_seed),
            )

            for job in episode_jobs:
                trigger_obs, _ = obs_schedule[int(job.trigger_obs_index)]
                _, trigger_proprio, _ = _encode_obs_cpu(
                    trigger_obs,
                    cfg=cfg,
                    processor=processor,
                    width=input_w,
                    height=input_h,
                )
                pred_chunk = replay_one_job_counterfactual(
                    job=job,
                    experiment=experiment,
                    action_model=action_model,
                    action_context=action_context,
                    action_context_mask=action_context_mask,
                    trigger_proprio=trigger_proprio,
                    clean_bank=clean_bank,
                    wrong_bank=wrong_bank,
                    action_horizon=action_horizon,
                    num_inference_steps=num_inference_steps,
                    sigma_shift=sigma_shift,
                    rand_device=rand_device,
                    seed=(None if args.seed is None else int(args.seed) + int(job.job_id)),
                    processor=processor,
                    cfg=cfg,
                )
                metrics = _compute_chunk_metrics(
                    pred_chunk=pred_chunk,
                    gt_actions=gt_actions_env,
                    trigger_env_step=int(job.trigger_env_step),
                )
                job_required_wrong_obs_indices = _get_required_wrong_obs_indices_for_job(
                    job=job,
                    experiment=experiment,
                )
                episode_results.append(
                    {
                        "episode_idx": int(episode_idx),
                        "experiment": str(experiment.name),
                        "job_id": int(job.job_id),
                        "trigger_env_step": int(job.trigger_env_step),
                        "trigger_obs_index": int(job.trigger_obs_index),
                        "normalized_trigger_obs_index": int(job.trigger_obs_index) - int(formal_obs_index_start),
                        "required_wrong_obs_indices": sorted(int(v) for v in job_required_wrong_obs_indices),
                        "normalized_required_wrong_obs_indices": sorted(
                            int(v) - int(formal_obs_index_start) for v in job_required_wrong_obs_indices
                        ),
                        **metrics,
                    }
                )
                all_result_rows.append(dict(episode_results[-1]))

        episode_payload: dict[str, Any] = {
            "episode_idx": int(episode_idx),
            "trial_idx": int(resolved_trial_idx),
            "gt_episode_index": int(gt_episode_index),
            "task_description": str(task_description),
            "formal_obs_index_start": int(formal_obs_index_start),
            "num_jobs": int(len(episode_jobs)),
            "experiment_summaries": _summarize_result_rows(episode_results, experiments=experiments),
        }
        if bool(args.save_job_results):
            episode_payload["results"] = episode_results
        results["episodes"].append(episode_payload)

    results["experiment_summaries"] = _summarize_result_rows(all_result_rows, experiments=experiments)

    output_path = Path(args.output_json).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    print(f"Wrote counterfactual replay results to: {output_path}")


if __name__ == "__main__":
    main()
