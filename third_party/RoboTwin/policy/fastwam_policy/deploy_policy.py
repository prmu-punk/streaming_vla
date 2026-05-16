from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from PIL import Image

def _find_project_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "src" / "fastwam").is_dir() and (parent / "configs").is_dir():
            return parent
    raise RuntimeError(f"Failed to locate FastWAM project root from: {current}")


PROJECT_ROOT = _find_project_root()
SRC_ROOT = PROJECT_ROOT / "src"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fastwam.datasets.lerobot.processors.fastwam_processor import FastWAMProcessor
from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
from fastwam.utils.async_streaming_runtime import ProfiledRuntime, StreamingRuntime

logger = logging.getLogger(__name__)

_LAST_POLICY: Optional["StreamingRuntimeRobotWinPolicy"] = None


def get_last_policy() -> Optional["StreamingRuntimeRobotWinPolicy"]:
    return _LAST_POLICY


def _is_none_like(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"", "none", "null"}
    return False


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y"}:
            return True
        if lowered in {"0", "false", "no", "n"}:
            return False
    raise ValueError(f"Cannot parse bool value: {value}")


def _parse_optional_int(value: Any) -> Optional[int]:
    if _is_none_like(value):
        return None
    return int(value)


def _parse_optional_float(value: Any) -> Optional[float]:
    if _is_none_like(value):
        return None
    return float(value)


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


def _resolve_sim_cfg_name(sim_cfg_path: Optional[str], sim_cfg_name: Optional[str]) -> str:
    configs_root = (PROJECT_ROOT / "configs").resolve()
    if not _is_none_like(sim_cfg_path):
        cfg_path = Path(str(sim_cfg_path)).expanduser().resolve()
        try:
            relative = cfg_path.relative_to(configs_root)
        except ValueError as exc:
            raise ValueError(
                f"`sim_cfg_path` must be under {configs_root}, got: {cfg_path}"
            ) from exc
        return relative.as_posix()

    if _is_none_like(sim_cfg_name):
        return "sim_robotwin.yaml"
    return str(sim_cfg_name)


def _compose_sim_cfg(
    sim_cfg_path: Optional[str],
    sim_cfg_name: Optional[str],
    sim_task: Optional[str],
) -> DictConfig:
    config_name = _resolve_sim_cfg_name(sim_cfg_path=sim_cfg_path, sim_cfg_name=sim_cfg_name)
    configs_root = (PROJECT_ROOT / "configs").resolve()
    overrides = []
    if not _is_none_like(sim_task):
        overrides.append(f"task={str(sim_task)}")

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    with initialize_config_dir(version_base="1.3", config_dir=str(configs_root)):
        cfg = compose(config_name=config_name, overrides=overrides)
    return cfg


def _resolve_dataset_stats_path(dataset_stats_path: Optional[str]) -> Path:
    if _is_none_like(dataset_stats_path):
        raise FileNotFoundError(
            "`dataset_stats_path` is required. "
            "Please pass it from eval entrypoint overrides."
        )
    resolved = Path(str(dataset_stats_path)).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Dataset stats path not found: {resolved}")
    return resolved


def _resize_rgb(image: np.ndarray, size_wh: tuple[int, int]) -> np.ndarray:
    pil_image = Image.fromarray(image.astype(np.uint8), mode="RGB")
    resized = pil_image.resize(size_wh, resample=Image.BILINEAR)
    return np.asarray(resized, dtype=np.uint8)


def _summarize_ms(samples_ms: list[float]) -> dict[str, float | int | None]:
    if len(samples_ms) == 0:
        return {"count": 0, "avg_ms": None, "p50_ms": None, "p90_ms": None, "max_ms": None}
    arr = np.asarray(samples_ms, dtype=np.float64)
    return {
        "count": int(arr.shape[0]),
        "avg_ms": float(np.mean(arr)),
        "p50_ms": float(np.percentile(arr, 50)),
        "p90_ms": float(np.percentile(arr, 90)),
        "max_ms": float(np.max(arr)),
    }


class StreamingRuntimeRobotWinPolicy:
    def __init__(
        self,
        model_cfg: DictConfig,
        processor_cfg: DictConfig,
        checkpoint_path: str,
        dataset_stats_path: Path,
        video_device: str,
        action_device: str,
        model_dtype: torch.dtype,
        action_horizon: int,
        replan_steps: int,
        num_inference_steps: int,
        sigma_shift: Optional[float],
        seed: Optional[int],
        text_cfg_scale: float,
        negative_prompt: str,
        rand_device: str,
        tiled: bool,
        timing_enabled: bool,
        profile_runtime: bool,
        video_layers_per_chunk: int,
    ) -> None:
        model_cfg_copy = OmegaConf.create(OmegaConf.to_container(model_cfg, resolve=True))
        model_cfg_copy.load_text_encoder = True

        self.model = instantiate(model_cfg_copy, model_dtype=model_dtype, device=video_device)
        self.model.load_checkpoint(checkpoint_path)
        self.model = self.model.to(video_device).eval()

        if str(action_device) == str(video_device):
            self.action_model = self.model
        else:
            self.action_model = instantiate(model_cfg_copy, model_dtype=model_dtype, device=action_device)
            self.action_model.load_checkpoint(checkpoint_path)
            self.action_model = self.action_model.to(action_device).eval()

        self.processor: FastWAMProcessor = instantiate(processor_cfg).eval()
        dataset_stats = load_dataset_stats_from_json(str(dataset_stats_path))
        self.processor.set_normalizer_from_stats(dataset_stats)

        self.action_horizon = int(action_horizon)
        self.replan_steps = int(max(1, min(replan_steps, action_horizon)))
        self.num_inference_steps = int(num_inference_steps)
        self.sigma_shift = sigma_shift
        self.seed = seed
        self.text_cfg_scale = float(text_cfg_scale)
        self.negative_prompt = str(negative_prompt)
        self.rand_device = str(rand_device)
        self.tiled = bool(tiled)
        self.timing_enabled = bool(timing_enabled)
        self.profile_runtime = bool(profile_runtime)
        self.video_layers_per_chunk = int(max(1, video_layers_per_chunk))

        self.episode_count = 0
        self.step_count = 0
        self._timing_rollout = {"infer_s": 0.0, "sim_s": 0.0}
        self._runtime: Optional[StreamingRuntime] = None
        self._runtime_started = False
        self._current_instruction: Optional[str] = None
        self._env_step_samples_ms: list[float] = []
        self._episode_stats: list[dict[str, Any]] = []

        logger.info(
            "Initialized StreamingRuntimeRobotWinPolicy | ckpt=%s | stats=%s | horizon=%d | replan=%d | "
            "video_device=%s | action_device=%s | profile=%s",
            checkpoint_path,
            dataset_stats_path,
            self.action_horizon,
            self.replan_steps,
            video_device,
            action_device,
            self.profile_runtime,
        )

    def _normalize_state(self, state: np.ndarray) -> torch.Tensor:
        state_meta = self.processor.shape_meta["state"]
        if len(state_meta) != 1:
            raise ValueError("Expected exactly one merged state key in shape_meta['state'].")
        state_key = state_meta[0]["key"]

        state_batch = {"state": {state_key: torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)}}
        state_batch = self.processor.action_state_transform(state_batch)
        state_batch = self.processor.normalizer.forward(state_batch)
        return state_batch["state"][state_key]

    def _denormalize_action(self, action: torch.Tensor) -> np.ndarray:
        if action.ndim == 2:
            action = action.unsqueeze(0)
        if action.ndim != 3:
            raise ValueError(f"Expected action tensor [B,T,D], got {tuple(action.shape)}")

        action_meta = self.processor.shape_meta["action"]
        if len(action_meta) != 1:
            raise ValueError("Expected exactly one merged action key in shape_meta['action'].")

        action_key = action_meta[0]["key"]
        normalizer = self.processor.normalizer.normalizers["action"][action_key]
        denorm = normalizer.backward(action.to(dtype=torch.float32, device="cpu"))
        return denorm.numpy()

    def _build_robotwin_image_tensor(self, observation: Dict[str, Any]) -> torch.Tensor:
        obs_data = observation["observation"]
        head = _resize_rgb(obs_data["head_camera"]["rgb"], (320, 256))
        left = _resize_rgb(obs_data["left_camera"]["rgb"], (160, 128))
        right = _resize_rgb(obs_data["right_camera"]["rgb"], (160, 128))
        bottom = np.concatenate([left, right], axis=1)
        image = np.concatenate([head, bottom], axis=0)  # [384, 320, 3]

        image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(dtype=torch.float32)
        image_tensor = image_tensor * (2.0 / 255.0) - 1.0
        return image_tensor

    def _encode_observation(self, observation: Dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        image_tensor = self._build_robotwin_image_tensor(observation)
        state_vector = np.asarray(observation["joint_action"]["vector"], dtype=np.float32)
        proprio = self._normalize_state(state_vector)
        return image_tensor, proprio

    def _action_postprocess(self, actions: torch.Tensor) -> np.ndarray:
        return self._denormalize_action(actions)[0]

    def _build_runtime(self, instruction: str) -> None:
        prompt = DEFAULT_PROMPT.format(task=instruction)
        with torch.no_grad():
            video_context, video_context_mask = self.model.encode_prompt(prompt)
            if self.action_model is self.model:
                action_context, action_context_mask = video_context, video_context_mask
            else:
                action_context, action_context_mask = self.action_model.encode_prompt(prompt)
        video_context = video_context.to(device="cpu", dtype=self.model.torch_dtype)
        video_context_mask = video_context_mask.to(device="cpu", dtype=torch.bool)
        action_context = action_context.to(device="cpu", dtype=self.action_model.torch_dtype)
        action_context_mask = action_context_mask.to(device="cpu", dtype=torch.bool)

        runtime_cls = ProfiledRuntime if self.profile_runtime else StreamingRuntime
        self._runtime = runtime_cls(
            video_model=self.model,
            action_model=self.action_model,
            video_context=video_context,
            video_context_mask=video_context_mask,
            action_context=action_context,
            action_context_mask=action_context_mask,
            action_postprocess=self._action_postprocess,
            action_horizon=self.action_horizon,
            num_inference_steps=self.num_inference_steps,
            sigma_shift=self.sigma_shift,
            rand_device=self.rand_device,
            tiled=self.tiled,
            action_trigger_every_n_obs=1,
            video_layers_per_chunk=self.video_layers_per_chunk,
            seed=self.seed,
        )
        self._runtime.start()
        self._runtime_started = True

    def _wait_for_action(self, env_step: int) -> np.ndarray:
        if self._runtime is None:
            raise RuntimeError("Streaming runtime is not initialized.")

        action = self._runtime.get_action(env_step)
        while action is None:
            self._runtime.wait_until_idle()
            action = self._runtime.get_action(env_step, count_miss=False)
        return np.asarray(action, dtype=np.float32)

    def _teardown_runtime(self) -> None:
        if self._runtime is None or not self._runtime_started:
            self._runtime = None
            self._runtime_started = False
            return

        try:
            self._runtime.stop()
        except Exception:
            logger.exception("Failed to stop streaming runtime cleanly.")

        try:
            stats = self._runtime.stats()
        except Exception:
            logger.exception("Failed to collect streaming runtime stats.")
            stats = None

        if stats is not None:
            stats = dict(stats)
            stats.setdefault("timing_ms", {})
            stats["timing_ms"]["env_step"] = _summarize_ms(self._env_step_samples_ms)
            stats.setdefault("timing_samples_ms", {})
            stats["timing_samples_ms"]["env_step"] = [float(v) for v in self._env_step_samples_ms]
            completed_steps = int(stats.get("completed_steps", 0))
            stats.setdefault("submitted_jobs", completed_steps)
            stats.setdefault("completed_jobs", completed_steps)
            self._episode_stats.append(stats)

        self._runtime = None
        self._runtime_started = False
        self._env_step_samples_ms = []

    def should_request_observation(self) -> bool:
        return True

    def step(self, task_env, observation: Optional[Dict[str, Any]]) -> None:
        if observation is None:
            raise ValueError("RobotWin streaming runtime requires an observation every take_action step.")

        env_step = int(self.step_count)
        if self._runtime is None:
            instruction = task_env.get_instruction()
            self._current_instruction = instruction
            self._build_runtime(instruction)

        if self._runtime is None:
            raise RuntimeError("Streaming runtime failed to initialize.")

        image_tensor, proprio = self._encode_observation(observation)

        infer_t0 = time.perf_counter() if self.timing_enabled else 0.0
        self._runtime.submit_observation(
            input_image=image_tensor,
            proprio=proprio,
            env_step=env_step,
            obs_index=env_step,
            obs_timestamp_ms=float(env_step),
            trigger_job=False,
        )
        action = self._wait_for_action(env_step)
        if self.timing_enabled:
            self._timing_rollout["infer_s"] += time.perf_counter() - infer_t0

        sim_t0 = time.perf_counter() if self.timing_enabled else 0.0
        step_t0 = time.perf_counter()
        task_env.take_action(action, action_type="qpos")
        if self.timing_enabled:
            self._timing_rollout["sim_s"] += time.perf_counter() - sim_t0
        self._env_step_samples_ms.append(float((time.perf_counter() - step_t0) * 1000.0))
        self.step_count += 1

    def reset_timing_rollout(self) -> None:
        self._timing_rollout["infer_s"] = 0.0
        self._timing_rollout["sim_s"] = 0.0

    def get_timing_rollout(self) -> Dict[str, float]:
        return {
            "infer_s": float(self._timing_rollout["infer_s"]),
            "sim_s": float(self._timing_rollout["sim_s"]),
        }

    def collect_episode_stats(self) -> list[dict[str, Any]]:
        self._teardown_runtime()
        return list(self._episode_stats)

    def reset(self) -> None:
        self._teardown_runtime()
        self.episode_count += 1
        self.step_count = 0
        self._current_instruction = None
        self._env_step_samples_ms = []
        self.reset_timing_rollout()


def encode_obs(observation: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    return observation


def get_model(usr_args: Dict[str, Any]):
    sim_cfg_path = usr_args.get("sim_cfg_path")
    sim_cfg_name = usr_args.get("sim_cfg_name")
    sim_task = usr_args.get("sim_task")
    cfg = _compose_sim_cfg(
        sim_cfg_path=sim_cfg_path,
        sim_cfg_name=sim_cfg_name,
        sim_task=sim_task,
    )

    checkpoint_path = usr_args.get("ckpt_setting")
    if _is_none_like(checkpoint_path):
        raise ValueError("`ckpt_setting` is required and must be a valid checkpoint path.")

    video_device = str(
        usr_args.get("async_video_device")
        or usr_args.get("video_device")
        or usr_args.get("device")
        or cfg.EVALUATION.get("device")
        or "cuda"
    )
    action_device = str(usr_args.get("async_action_device") or usr_args.get("action_device") or video_device)
    if video_device.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA is unavailable; fallback both runtime devices to cpu.")
        video_device = "cpu"
        action_device = "cpu"

    mixed_precision = str(usr_args.get("mixed_precision") or cfg.get("mixed_precision", "bf16"))
    model_dtype = _mixed_precision_to_model_dtype(mixed_precision)

    dataset_stats_path = _resolve_dataset_stats_path(
        dataset_stats_path=usr_args.get("dataset_stats_path"),
    )

    action_horizon = _parse_optional_int(usr_args.get("action_horizon"))
    if action_horizon is None:
        eval_horizon = _parse_optional_int(cfg.EVALUATION.get("action_horizon"))
        action_horizon = eval_horizon if eval_horizon is not None else int(cfg.data.train.num_frames) - 1
    if action_horizon <= 0:
        raise ValueError(f"`action_horizon` must be positive, got {action_horizon}")

    replan_steps = _parse_optional_int(usr_args.get("replan_steps"))
    if replan_steps is None:
        replan_steps = int(cfg.EVALUATION.get("replan_steps", 8))

    num_inference_steps = _parse_optional_int(usr_args.get("num_inference_steps"))
    if num_inference_steps is None:
        num_inference_steps = int(cfg.EVALUATION.get("num_inference_steps", cfg.eval_num_inference_steps))

    sigma_shift = _parse_optional_float(usr_args.get("sigma_shift"))
    if sigma_shift is None:
        sigma_shift = _parse_optional_float(cfg.EVALUATION.get("sigma_shift"))

    seed = _parse_optional_int(usr_args.get("seed"))
    text_cfg_scale = float(usr_args.get("text_cfg_scale", cfg.EVALUATION.get("text_cfg_scale", 1.0)))
    negative_prompt = str(usr_args.get("negative_prompt", cfg.EVALUATION.get("negative_prompt", "")))
    rand_device = str(usr_args.get("rand_device", cfg.EVALUATION.get("rand_device", "cpu")))
    tiled = _parse_bool(usr_args.get("tiled", cfg.EVALUATION.get("tiled", False)))
    timing_enabled = _parse_bool(
        usr_args.get("timing_enabled", cfg.EVALUATION.get("timing_enabled", False))
    )
    profile_runtime = _parse_bool(usr_args.get("profile_runtime", False))
    video_layers_per_chunk = _parse_optional_int(usr_args.get("async_video_layers_per_chunk"))
    if video_layers_per_chunk is None:
        video_layers_per_chunk = int(getattr(cfg.model.video_dit_config, "num_layers", 30))

    OmegaConf.set_struct(cfg, False)
    cfg.model.load_text_encoder = True
    if not _is_none_like(usr_args.get("redirect_common_files")):
        cfg.model.redirect_common_files = _parse_bool(usr_args.get("redirect_common_files"))
    OmegaConf.set_struct(cfg, True)

    policy = StreamingRuntimeRobotWinPolicy(
        model_cfg=cfg.model,
        processor_cfg=cfg.data.train.processor,
        checkpoint_path=str(checkpoint_path),
        dataset_stats_path=dataset_stats_path,
        video_device=video_device,
        action_device=action_device,
        model_dtype=model_dtype,
        action_horizon=action_horizon,
        replan_steps=replan_steps,
        num_inference_steps=num_inference_steps,
        sigma_shift=sigma_shift,
        seed=seed,
        text_cfg_scale=text_cfg_scale,
        negative_prompt=negative_prompt,
        rand_device=rand_device,
        tiled=tiled,
        timing_enabled=timing_enabled,
        profile_runtime=profile_runtime,
        video_layers_per_chunk=video_layers_per_chunk,
    )
    global _LAST_POLICY
    _LAST_POLICY = policy
    return policy


def eval(TASK_ENV, model, observation: Optional[Dict[str, Any]]):
    obs = encode_obs(observation)
    model.step(TASK_ENV, obs)


def reset_model(model):
    model.reset()
