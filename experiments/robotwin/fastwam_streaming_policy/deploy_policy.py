"""
Streaming variant of `experiments/robotwin/fastwam_policy/deploy_policy.py`.

Drives the `AsyncStreamingActionRuntimeProfiled` from inside a RoboTwin-shaped 
policy step. Designed to be plugged into the real `third_party/RoboTwin/script/eval_policy.py` 
via the `policy/fastwam_streaming_policy` symlink.

The policy is deliberately sim-agnostic: every input it touches comes from the
existing non-streaming `WorldActionRobotWinPolicy` (image composition, state
normalization, action denormalization).
"""
from __future__ import annotations

import logging
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from experiments.libero.async_streaming_runtime_profiled import (  # noqa: E402
    AsyncStreamingActionRuntimeProfiled,
)
from experiments.robotwin.fastwam_policy.deploy_policy import (  # noqa: E402
    WorldActionRobotWinPolicy,
    _compose_sim_cfg,
    _is_none_like,
    _mixed_precision_to_model_dtype,
    _parse_bool,
    _parse_optional_float,
    _parse_optional_int,
    _resolve_dataset_stats_path,
)
from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT  # noqa: E402

logger = logging.getLogger(__name__)

# Module-level handle so external profiling drivers (e.g. the real-machine
# entrypoint that delegates to RoboTwin's own `script/eval_policy.py`) can grab
# the built policy after `get_model` is invoked from inside RoboTwin's main().
_LAST_POLICY: Optional["StreamingWorldActionRobotWinPolicy"] = None


def get_last_policy() -> Optional["StreamingWorldActionRobotWinPolicy"]:
    return _LAST_POLICY


class StreamingWorldActionRobotWinPolicy(WorldActionRobotWinPolicy):
    """Streaming policy: replaces sync `infer_action` with async runtime."""

    def __init__(
        self,
        *,
        action_model_cfg: DictConfig,
        action_device: str,
        async_obs_stride_env_steps: int,
        async_action_trigger_every_n_obs: int,
        async_video_layers_per_chunk: int,
        async_force_first_job: bool,
        async_warmup_action_jobs: int,
        async_control_dt_ms: float,
        **kwargs: Any,
    ) -> None:
        # We deliberately do NOT call `super().__init__` because the parent
        # forces `model_cfg_copy.load_text_encoder = True`, which would
        # trigger a 10 GB T5 download on a fresh server. The streaming runtime
        # only needs prompt context tensors, not real text encoding — the
        # `_streaming_patches` module installs a fallback `encode_prompt`
        # that returns zeros when text_encoder is None. We therefore inline
        # the parent ctor and respect the user's `load_text_encoder` setting.
        from hydra.utils import instantiate
        from fastwam.datasets.lerobot.processors.fastwam_processor import (
            FastWAMProcessor,
        )
        from fastwam.datasets.lerobot.utils.normalizer import (
            load_dataset_stats_from_json,
        )

        ckpt_str = str(kwargs["checkpoint_path"])
        self._ckpt_path_str = ckpt_str

        model_cfg_copy = OmegaConf.create(
            OmegaConf.to_container(kwargs["model_cfg"], resolve=True)
        )
        # Respect whatever load_text_encoder the cfg already has (default false
        # in the streaming entrypoint to avoid the T5 download).
        self.model = instantiate(
            model_cfg_copy, model_dtype=kwargs["model_dtype"], device=kwargs["device"]
        )
        self.model.load_checkpoint(ckpt_str)
        self.model = self.model.to(kwargs["device"]).eval()

        self.processor: FastWAMProcessor = instantiate(kwargs["processor_cfg"]).eval()
        dataset_stats = load_dataset_stats_from_json(str(kwargs["dataset_stats_path"]))
        self.processor.set_normalizer_from_stats(dataset_stats)

        self.action_horizon = int(kwargs["action_horizon"])
        self.replan_steps = int(
            max(1, min(int(kwargs["replan_steps"]), int(kwargs["action_horizon"])))
        )
        self.num_inference_steps = int(kwargs["num_inference_steps"])
        self.sigma_shift = kwargs["sigma_shift"]
        self.seed = kwargs["seed"]
        self.text_cfg_scale = float(kwargs["text_cfg_scale"])
        self.negative_prompt = str(kwargs["negative_prompt"])
        self.rand_device = str(kwargs["rand_device"])
        self.tiled = bool(kwargs["tiled"])
        self.timing_enabled = bool(kwargs["timing_enabled"])
        self._num_video_frames = int(kwargs["num_video_frames"])

        self.episode_count = 0
        self.step_count = 0
        self._timing_rollout = {"infer_s": 0.0, "sim_s": 0.0}

        # Build the second (action-side) model on its own GPU.
        action_model = instantiate(
            action_model_cfg,
            model_dtype=self.model.torch_dtype,
            device=action_device,
        )
        action_model.load_checkpoint(ckpt_str)
        # Detach text_encoder before .to() so it stays on CPU; the patched
        # FastWAM.to already does this, but be explicit.
        if getattr(action_model, "text_encoder", None) is not None:
            try:
                action_model.text_encoder.to("cpu")
            except Exception:  # pragma: no cover - defensive
                pass
        self.action_model = action_model.to(action_device).eval()
        self._action_device = str(action_device)

        self._async_obs_stride_env_steps = int(async_obs_stride_env_steps)
        self._async_action_trigger_every_n_obs = int(async_action_trigger_every_n_obs)
        self._async_video_layers_per_chunk = int(async_video_layers_per_chunk)
        self._async_force_first_job = bool(async_force_first_job)
        self._async_warmup_action_jobs = int(async_warmup_action_jobs)
        self._async_control_dt_ms = float(async_control_dt_ms)

        self._runtime: Optional[AsyncStreamingActionRuntimeProfiled] = None
        self._runtime_started = False
        self._t_env: int = 0
        self._obs_counter: int = 0
        self._formal_obs_count: int = 0
        self._first_formal_triggered: bool = False
        self._last_action: Optional[np.ndarray] = None
        self._episode_stats: list[dict[str, Any]] = []
        self._current_instruction: Optional[str] = None

        # We do not use the parent's pending_actions deque; force-empty so
        # `should_request_observation()` always returns True.
        self.pending_actions: deque[np.ndarray] = deque()

    # The non-streaming parent stashes the ckpt path on the model side; we
    # need it again to build the second expert. Re-read it from the model.
    @property
    def _checkpoint_path(self) -> str:
        # `WorldActionRobotWinPolicy.__init__` doesn't store ckpt; rebuild via
        # the env arg passed in. Set lazily by classmethod factory below.
        return getattr(self, "_ckpt_path_str")

    # ---- RoboTwin contract ---------------------------------------------
    def should_request_observation(self) -> bool:
        return True  # streaming runtime ingests every obs

    def reset(self) -> None:
        # Mirrors parent.reset() bookkeeping but also tears down the runtime
        # so the next episode rebuilds it with a fresh prompt encoding.
        super().reset()
        self._teardown_runtime()
        self._t_env = 0
        self._obs_counter = 0
        self._formal_obs_count = 0
        self._first_formal_triggered = False
        self._last_action = None
        self._current_instruction = None

    def _teardown_runtime(self) -> None:
        if self._runtime is not None and self._runtime_started:
            try:
                self._runtime.stop()
            except Exception:
                logger.exception("Failed to stop streaming runtime cleanly.")
            stats = None
            try:
                stats = self._runtime.stats()
            except Exception:
                logger.exception("Failed to collect runtime stats.")
            if stats is not None:
                self._episode_stats.append(stats)
        self._runtime = None
        self._runtime_started = False

    def _build_runtime(self, instruction: str) -> None:
        prompt = DEFAULT_PROMPT.format(task=instruction)
        with torch.no_grad():
            v_ctx, v_mask = self.model.encode_prompt(prompt)
            a_ctx, a_mask = self.action_model.encode_prompt(prompt)
        v_ctx = v_ctx.to(device="cpu", dtype=self.model.torch_dtype)
        v_mask = v_mask.to(device="cpu", dtype=torch.bool)
        a_ctx = a_ctx.to(device="cpu", dtype=self.action_model.torch_dtype)
        a_mask = a_mask.to(device="cpu", dtype=torch.bool)

        action_postprocess = lambda x: self._denormalize_action(x)[0]

        self._runtime = AsyncStreamingActionRuntimeProfiled(
            video_model=self.model,
            action_model=self.action_model,
            video_context=v_ctx,
            video_context_mask=v_mask,
            action_context=a_ctx,
            action_context_mask=a_mask,
            action_postprocess=action_postprocess,
            action_horizon=self.action_horizon,
            num_inference_steps=self.num_inference_steps,
            sigma_shift=self.sigma_shift,
            rand_device=self.rand_device,
            tiled=self.tiled,
            action_trigger_every_n_obs=self._async_action_trigger_every_n_obs,
            video_layers_per_chunk=self._async_video_layers_per_chunk,
            seed=self.seed,
        )
        self._runtime.start()
        self._runtime_started = True
        self._runtime.reset_for_formal_phase(env_step=0)

    def _encode_observation(
        self, observation: Dict[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        image_tensor_dev = self._build_robotwin_image_tensor(observation)
        image_cpu = image_tensor_dev.detach().to(device="cpu", dtype=torch.float32)

        state_vector = np.asarray(
            observation["joint_action"]["vector"], dtype=np.float32
        )
        proprio = self._normalize_state(state_vector)
        return image_cpu, proprio

    def step(self, task_env, observation: Optional[Dict[str, Any]]) -> None:
        if observation is None:
            raise ValueError(
                "Streaming policy requires an observation every step "
                "(should_request_observation always returns True)."
            )

        instruction = task_env.get_instruction()
        if self._runtime is None:
            self._current_instruction = instruction
            self._build_runtime(instruction)

        image_cpu, proprio = self._encode_observation(observation)
        t = self._t_env

        if t % self._async_obs_stride_env_steps == 0:
            should_trigger = self._runtime.should_trigger_on_obs(
                self._formal_obs_count + 1
            )
            if self._async_force_first_job and not self._first_formal_triggered:
                should_trigger = True
            self._runtime.submit_observation(
                input_image=image_cpu,
                proprio=proprio,
                env_step=t,
                obs_index=self._obs_counter,
                obs_timestamp_ms=float(self._obs_counter)
                * self._async_control_dt_ms
                * float(self._async_obs_stride_env_steps),
                trigger_job=should_trigger,
            )
            self._obs_counter += 1
            self._formal_obs_count += 1
            if should_trigger:
                self._first_formal_triggered = True

        action = self._runtime.get_action(t)
        if action is None:
            action = (
                self._last_action
                if self._last_action is not None
                else np.zeros(14, dtype=np.float32)
            )

        sim_t0 = time.perf_counter() if self.timing_enabled else 0.0
        task_env.take_action(action, action_type="qpos")
        if self.timing_enabled:
            self._timing_rollout["sim_s"] += time.perf_counter() - sim_t0

        self._last_action = np.asarray(action, dtype=np.float32)
        self._t_env += 1
        self.step_count += 1

    # ---- final stats dump ----------------------------------------------
    def collect_episode_stats(self) -> list[dict[str, Any]]:
        # Ensure final episode (if any) is captured.
        self._teardown_runtime()
        return list(self._episode_stats)


# ---- get_model / eval / reset_model -----------------------------------------
def get_model(usr_args: Dict[str, Any]):
    sim_cfg_path = usr_args.get("sim_cfg_path")
    sim_cfg_name = usr_args.get("sim_cfg_name")
    sim_task = usr_args.get("sim_task")
    cfg = _compose_sim_cfg(
        sim_cfg_path=sim_cfg_path,
        sim_cfg_name=sim_cfg_name,
        sim_task=sim_task,
    )

    want_text_encoder = _parse_bool(usr_args.get("load_text_encoder", True))
    want_redirect = _parse_bool(usr_args.get("redirect_common_files", True))
    OmegaConf.set_struct(cfg, False)
    cfg.model.load_text_encoder = bool(want_text_encoder)
    cfg.model.redirect_common_files = bool(want_redirect)
    OmegaConf.set_struct(cfg, True)

    checkpoint_path = usr_args.get("ckpt_setting")
    if _is_none_like(checkpoint_path):
        raise ValueError("`ckpt_setting` is required and must be a valid checkpoint path.")
    ckpt_str = str(checkpoint_path)

    video_device = str(usr_args.get("async_video_device") or "cuda:0")
    action_device = str(usr_args.get("async_action_device") or "cuda:1")
    if video_device.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA unavailable; falling back to cpu for both experts.")
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
        action_horizon = (
            eval_horizon
            if eval_horizon is not None
            else int(cfg.data.train.num_frames) - 1
        )

    replan_steps = _parse_optional_int(usr_args.get("replan_steps"))
    if replan_steps is None:
        replan_steps = int(cfg.EVALUATION.get("replan_steps", 8))

    num_inference_steps = _parse_optional_int(usr_args.get("num_inference_steps"))
    if num_inference_steps is None:
        num_inference_steps = int(
            cfg.EVALUATION.get("num_inference_steps", cfg.eval_num_inference_steps)
        )

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

    async_obs_stride = int(usr_args.get("async_obs_stride_env_steps", 3))
    async_trigger = int(usr_args.get("async_action_trigger_every_n_obs", 3))
    async_layers = int(usr_args.get("async_video_layers_per_chunk", 2))
    async_force_first = _parse_bool(usr.args.get("async_force_first_job", True))
    async_warmup = int(usr.args.get("async_warmup_action_jobs", 0))
    async_dt_ms = float(usr.args.get("async_control_dt_ms", 50.0))

    policy = StreamingWorldActionRobotWinPolicy(
        # parent ctor args
        model_cfg=cfg.model,
        processor_cfg=cfg.data.train.processor,
        checkpoint_path=ckpt_str,
        dataset_stats_path=dataset_stats_path,
        device=video_device,
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
        num_video_frames=(int(cfg.data.train.num_frames) - 1)
        // int(cfg.data.train.action_video_freq_ratio)
        + 1,
        # streaming-only ctor args
        action_model_cfg=cfg.model,
        action_device=action_device,
        async_obs_stride_env_steps=async_obs_stride,
        async_action_trigger_every_n_obs=async_trigger,
        async_video_layers_per_chunk=async_layers,
        async_force_first_job=async_force_first,
        async_warmup_action_jobs=async_warmup,
        async_control_dt_ms=async_dt_ms,
    )
    # Stash ckpt path for the secondary expert build inside the ctor (it
    # already used `checkpoint_path` for the primary; we need the same string
    # for the action expert ckpt load).
    policy._ckpt_path_str = ckpt_str  # noqa: SLF001
    global _LAST_POLICY
    _LAST_POLICY = policy
    return policy


def encode_obs(observation: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    return observation


def eval(TASK_ENV, model, observation: Optional[Dict[str, Any]]):
    obs = encode_obs(observation)
    model.step(TASK_ENV, obs)


def reset_model(model):
    model.reset()
