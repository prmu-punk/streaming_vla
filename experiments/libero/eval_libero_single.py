import json
import inspect
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

import hydra
import numpy as np
import torch
from accelerate import PartialState
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from tqdm import tqdm

# try:
#     import rootutils

#     rootutils.setup_root(__file__, indicator=".python-version", pythonpath=True)
# except ModuleNotFoundError:
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from experiments.libero.libero_utils import (
    LIBERO_ENV_RESOLUTION,
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    invert_gripper_action,
    quat2axisangle,
    save_prediction_video,
    save_rollout_video,
)
from fastwam.datasets.lerobot.processors.fastwam_processor import FastWAMProcessor
from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
from fastwam.utils.pytorch_utils import set_global_seed
from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
from libero.libero import benchmark
from experiments.libero.action_ensembler import ActionEnsembler
from experiments.libero.async_streaming_runtime import AsyncStreamingActionRuntime

OmegaConf.register_new_resolver("eval", eval)
OmegaConf.register_new_resolver("max", lambda x: max(x))
OmegaConf.register_new_resolver("split", lambda s, idx: s.split("/")[int(idx)])

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


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


def _resolve_eval_device(cfg: DictConfig) -> str:
    eval_device = cfg.EVALUATION.get("device")
    if eval_device is not None:
        return str(eval_device)
    return "cuda" if torch.cuda.is_available() else "cpu"


def _resolve_async_runtime_devices(cfg: DictConfig, fallback_device: str) -> tuple[str, str]:
    video_device = cfg.EVALUATION.get("async_video_device")
    action_device = cfg.EVALUATION.get("async_action_device")
    resolved_video_device = fallback_device if video_device is None else str(video_device)
    resolved_action_device = fallback_device if action_device is None else str(action_device)
    return resolved_video_device, resolved_action_device


def _build_eval_model(cfg: DictConfig, *, model_dtype: torch.dtype, device: str) -> torch.nn.Module:
    model = instantiate(cfg.model, model_dtype=model_dtype, device=device)
    if cfg.get("ckpt") is not None:
        _load_model_checkpoint(model, str(cfg.ckpt))
    else:
        logging.warning("No checkpoint provided; using randomly initialized weights for rollout timing.")
    return model.to(device).eval()


def _configure_egl_device(cfg: DictConfig) -> int:
    gpu_id = int(cfg.get("gpu_id", 0))
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(gpu_id)
    return gpu_id


def _resolve_dataset_stats_path(cfg: DictConfig) -> Path:
    explicit = cfg.EVALUATION.get("dataset_stats_path")
    candidates: list[Path] = []

    if explicit is not None:
        candidates.append(Path(os.path.expanduser(os.path.expandvars(str(explicit)))))

    if cfg.get("ckpt") is not None:
        ckpt = Path(os.path.expanduser(os.path.expandvars(str(cfg.ckpt))))
        for parent in list(ckpt.parents)[:4]:
            candidates.append(parent / "dataset_stats.json")

    dataset_dirs = [str(v) for v in cfg.data.train.get("dataset_dirs", [])]
    if any("libero" in v for v in dataset_dirs):
        candidates.append(project_root / "checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json")
    elif any("robotwin" in v for v in dataset_dirs):
        candidates.append(project_root / "checkpoints/fastwam_release/robotwin_uncond_3cam_384_dataset_stats.json")

    seen = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.exists():
            return resolved

    msg = (
        "Failed to locate dataset_stats.json. Tried explicit "
        "EVALUATION.dataset_stats_path and checkpoint parent directories. "
        "Please pass EVALUATION.dataset_stats_path=/path/to/dataset_stats.json."
    )
    raise FileNotFoundError(msg)


def _load_model_checkpoint(model: torch.nn.Module, ckpt: str) -> None:
    model.load_checkpoint(ckpt)
    logging.info("Loaded checkpoint via model.load_checkpoint: %s", ckpt)
    return


def _center_crop_resize(image: np.ndarray, width: int, height: int) -> np.ndarray:
    pil_image = Image.fromarray(image)
    src_w, src_h = pil_image.size
    scale = max(width / src_w, height / src_h)
    resized = pil_image.resize((round(src_w * scale), round(src_h * scale)), resample=Image.BILINEAR)
    rw, rh = resized.size
    left = max((rw - width) // 2, 0)
    top = max((rh - height) // 2, 0)
    cropped = resized.crop((left, top, left + width, top + height))
    return np.asarray(cropped, dtype=np.uint8)


def _normalize_proprio(
    proprio: np.ndarray,
    processor: FastWAMProcessor,
) -> torch.Tensor:
    state_meta = processor.shape_meta["state"]
    if len(state_meta) != 1:
        raise ValueError(
            "LIBERO eval currently expects a single merged state key in shape_meta['state']."
        )
    state_key = state_meta[0]["key"]

    state_batch = {"state": {state_key: torch.as_tensor(proprio, dtype=torch.float32).unsqueeze(0)}}
    state_batch = processor.action_state_transform(state_batch)
    state_batch = processor.normalizer.forward(state_batch)
    return state_batch["state"][state_key]


def _obs_to_model_input(
    obs: dict,
    cfg: DictConfig,
    processor: FastWAMProcessor,
    width: int,
    height: int,
    device: str,
    dtype: torch.dtype,
):
    imgs = get_libero_image(obs)
    image_meta = processor.shape_meta["images"]
    if len(image_meta) < int(processor.num_output_cameras):
        raise ValueError(
            f"shape_meta.images has {len(image_meta)} entries, "
            f"but num_output_cameras={processor.num_output_cameras}."
        )

    def _meta_to_hw(meta: dict, camera_idx: int) -> tuple[int, int]:
        shape = meta["shape"]
        if len(shape) != 3:
            raise ValueError(f"shape_meta.images[{camera_idx}].shape must be [C,H,W], got {shape}")
        return int(shape[1]), int(shape[2])

    concatenation = cfg.data.train.get("concat_multi_camera", "horizontal")
    num_cameras = processor.num_output_cameras
    if num_cameras == 1:
        primary_h, primary_w = _meta_to_hw(image_meta[0], camera_idx=0)
        rgb = _center_crop_resize(imgs["image"], width=primary_w, height=primary_h)
    elif num_cameras == 2:
        primary_h, primary_w = _meta_to_hw(image_meta[0], camera_idx=0)
        wrist_h, wrist_w = _meta_to_hw(image_meta[1], camera_idx=1)
        primary = _center_crop_resize(imgs["image"], width=primary_w, height=primary_h)
        wrist = _center_crop_resize(imgs["wrist_image"], width=wrist_w, height=wrist_h)
        if concatenation == "horizontal":
            rgb = np.concatenate([primary, wrist], axis=1)
        elif concatenation == "vertical":
            rgb = np.concatenate([primary, wrist], axis=0)
        else:
            raise ValueError(f"Invalid concat_multi_camera: {concatenation}")
    else:
        raise ValueError(f"LIBERO eval currently supports num_output_cameras in [1, 2], got {num_cameras}.")

    actual_h, actual_w = int(rgb.shape[0]), int(rgb.shape[1])
    expected_h, expected_w = int(height), int(width)
    image_shapes = [meta["shape"] for meta in image_meta]
    assert actual_h == expected_h and actual_w == expected_w, (
        "Input image size mismatch after per-camera resize + concat: "
        f"got (H,W)=({actual_h},{actual_w}), expected (H,W)=({expected_h},{expected_w}) "
        f"from data.train.video_size={[expected_h, expected_w]}; "
        f"shape_meta.images={image_shapes}, concat_multi_camera={concatenation}."
    )

    x = torch.tensor(rgb).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=dtype)
    x = x * (2.0 / 255.0) - 1.0

    proprio = _normalize_proprio(_extract_sim_state(obs), processor)

    return x, proprio, imgs


def _extract_sim_state(obs: dict) -> np.ndarray:
    """Build simulator state from current observation.

    This is used as proprio input for model inference.
    """
    state = np.concatenate(
        (
            obs["robot0_eef_pos"],
            quat2axisangle(obs["robot0_eef_quat"]),
            obs["robot0_gripper_qpos"],
        )
    ).astype(np.float32)
    return state


def _denormalize_action(action: torch.Tensor, processor: FastWAMProcessor) -> np.ndarray:
    if action.ndim == 2:
        action = action.unsqueeze(0)
    if action.ndim != 3:
        raise ValueError(f"Expected action tensor [B, T, D], got {tuple(action.shape)}")

    action_meta = processor.shape_meta["action"]
    if len(action_meta) != 1:
        raise ValueError(
            "LIBERO eval currently expects a single merged action key in shape_meta['action']."
        )

    action_key = action_meta[0]["key"]
    normalizer = processor.normalizer.normalizers["action"][action_key]
    action = action.to(dtype=torch.float32, device="cpu")
    denorm = normalizer.backward(action)
    return denorm.numpy()


def _get_num_video_frames(cfg: DictConfig) -> int:
    return (int(cfg.data.train.num_frames) - 1) // int(cfg.data.train.action_video_freq_ratio) + 1


def _validate_visualize_future_video_cfg(cfg: DictConfig) -> None:
    if not bool(cfg.EVALUATION.get("visualize_future_video", False)):
        return

    action_conditioned = cfg.model.video_dit_config.get("action_conditioned", None)
    if action_conditioned is not False:
        raise ValueError(
            "EVALUATION.visualize_future_video=true requires "
            "model.video_dit_config.action_conditioned=false."
        )


def _select_predicted_future_frames(pred_video: list[Image.Image], cfg: DictConfig) -> list[Image.Image]:
    if len(pred_video) == 0:
        raise ValueError("`infer_joint` returned an empty predicted video.")

    replan_steps = int(cfg.EVALUATION.get("replan_steps", 5))
    action_video_freq_ratio = int(cfg.data.train.action_video_freq_ratio)
    num_future_frames = replan_steps // action_video_freq_ratio
    keep_frames = 1 + num_future_frames
    return list(pred_video[:keep_frames])


def _get_future_frame_capture_steps(cfg: DictConfig) -> list[int]:
    replan_steps = int(cfg.EVALUATION.get("replan_steps", 5))
    action_video_freq_ratio = int(cfg.data.train.action_video_freq_ratio)
    num_future_frames = replan_steps // action_video_freq_ratio
    return [step_idx * action_video_freq_ratio for step_idx in range(num_future_frames + 1)]


def _frame_to_rgb_array(frame: Any) -> np.ndarray:
    if isinstance(frame, dict):
        images = []
        for value in frame.values():
            value_array = np.array(value) if isinstance(value, Image.Image) else np.array(value, copy=True)
            images.append(value_array)
        return np.concatenate(images, axis=1)
    if isinstance(frame, Image.Image):
        return np.array(frame.convert("RGB"))
    return np.array(frame, copy=True)


def _compute_clip_mean_psnr(
    gt_frames: list[Any],
    pred_frames: list[Any],
    eps: float = 1e-8,
) -> Optional[float]:
    if len(gt_frames) == 0 or len(pred_frames) == 0:
        return None
    assert len(gt_frames) == len(pred_frames), (
        "GT/pred frame count mismatch for PSNR: "
        f"len(gt_frames)={len(gt_frames)} len(pred_frames)={len(pred_frames)}. "
        "This indicates temporal misalignment in future-video capture."
    )
    num_frames = len(gt_frames)

    frame_psnr_values = []
    for gt_frame, pred_frame in zip(gt_frames[:num_frames], pred_frames[:num_frames]):
        gt_image = _frame_to_rgb_array(gt_frame)
        pred_image = _frame_to_rgb_array(pred_frame)
        target_h, target_w = pred_image.shape[:2]
        if gt_image.shape[:2] != (target_h, target_w):
            gt_image = np.array(
                Image.fromarray(gt_image).resize((target_w, target_h), resample=Image.BILINEAR)
            )

        gt_f32 = gt_image.astype(np.float32)
        pred_f32 = pred_image.astype(np.float32)
        mse = float(np.mean((pred_f32 - gt_f32) ** 2))
        psnr = 10.0 * np.log10((255.0 * 255.0) / max(mse, eps))
        frame_psnr_values.append(float(psnr))

    if len(frame_psnr_values) == 0:
        return None
    return float(np.mean(frame_psnr_values))


def _predict_action_chunk(
    obs: dict,
    task_description: str,
    model: torch.nn.Module,
    processor: FastWAMProcessor,
    cfg: DictConfig,
    *,
    action_horizon: int,
    input_w: int,
    input_h: int,
    model_device: str,
) -> tuple[np.ndarray, dict, Optional[list[Image.Image]]]:
    num_inference_steps_cfg = cfg.EVALUATION.get("num_inference_steps", None)
    if num_inference_steps_cfg is None:
        num_inference_steps = int(cfg.get("eval_num_inference_steps", 20))
    else:
        num_inference_steps = int(num_inference_steps_cfg)
    prompt_template = DEFAULT_PROMPT
    prompt = prompt_template.format(task=task_description)

    image, proprio, imgs = _obs_to_model_input(
        obs,
        cfg=cfg,
        processor=processor,
        width=input_w,
        height=input_h,
        device=model_device,
        dtype=model.torch_dtype,
    )

    infer_kwargs = {
        "prompt": prompt,
        "input_image": image,
        "action_horizon": action_horizon,
        "negative_prompt": str(cfg.EVALUATION.get("negative_prompt", "")),
        "text_cfg_scale": float(cfg.EVALUATION.get("text_cfg_scale", 1.0)),
        "num_inference_steps": num_inference_steps,
        "proprio": proprio,
        "sigma_shift": (
            None
            if cfg.EVALUATION.get("sigma_shift") is None
            else float(cfg.EVALUATION.get("sigma_shift"))
        ),
        "seed": None if cfg.get("seed") is None else int(cfg.seed),
        "rand_device": str(cfg.EVALUATION.get("rand_device", "cpu")),
        "tiled": bool(cfg.EVALUATION.get("tiled", False)),
    }
    visualize_future_video = bool(cfg.EVALUATION.get("visualize_future_video", False))
    predicted_future_frames = None
    if visualize_future_video:
        infer_kwargs["num_video_frames"] = _get_num_video_frames(cfg)
    elif "num_video_frames" in inspect.signature(model.infer_action).parameters:
        infer_kwargs["num_video_frames"] = _get_num_video_frames(cfg)

    with torch.no_grad():
        if visualize_future_video:
            pred = model.infer_joint(**infer_kwargs)
            predicted_future_frames = _select_predicted_future_frames(pred["video"], cfg)
        else:
            pred = model.infer_action(**infer_kwargs)
    action = pred["action"]  # [T, D]

    action = _denormalize_action(action, processor)[0]  # [T, D]

    # The dataloader flips the sign of the gripper action to align with other datasets
    # (0 = close, 1 = open), so flip it back (-1 = open, +1 = close) before executing the action
    action[..., -1] = action[..., -1] * 2 - 1
    action = invert_gripper_action(action)
    if bool(cfg.EVALUATION.get("binarize_gripper", False)):
        action[..., -1] = np.sign(action[..., -1])
    return action, imgs, predicted_future_frames


def _postprocess_libero_action_chunk(
    action_latents: torch.Tensor,
    *,
    processor: FastWAMProcessor,
    cfg: DictConfig,
) -> np.ndarray:
    action = _denormalize_action(action_latents, processor)[0]
    action[..., -1] = action[..., -1] * 2 - 1
    action = invert_gripper_action(action)
    if bool(cfg.EVALUATION.get("binarize_gripper", False)):
        action[..., -1] = np.sign(action[..., -1])
    return action


def _get_max_steps(task_suite_name: str) -> int:
    suite_steps = {
        "libero_spatial": 400,
        "libero_object": 400,
        "libero_goal": 400,
        "libero_10": 700,
        "libero_90": 700,
    }
    if task_suite_name not in suite_steps:
        raise ValueError(f"Unknown task suite: {task_suite_name}")
    return suite_steps[task_suite_name]


def _summarize_async_runtime_episodes(episodes: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
    if len(episodes) == 0:
        return None

    scalar_keys = [
        "submitted_obs",
        "submitted_jobs",
        "completed_jobs",
        "actions_served",
        "actions_missed",
        "dropped_prefix_actions",
    ]
    summary: dict[str, Any] = {"num_episodes": int(len(episodes))}
    for key in scalar_keys:
        values = [float(ep[key]) for ep in episodes]
        summary[f"{key}_total"] = float(np.sum(values))
        summary[f"{key}_mean"] = float(np.mean(values))

    timing_keys = ["video_refresh", "action_job", "action_step", "snapshot_copy"]
    timing_summary: dict[str, Any] = {}
    for key in timing_keys:
        counts = [
            int(ep.get("timing_ms", {}).get(key, {}).get("count", 0))
            for ep in episodes
        ]
        avg_values = [
            ep.get("timing_ms", {}).get(key, {}).get("avg_ms")
            for ep in episodes
        ]
        weighted_sum = 0.0
        total_count = 0
        for count, avg_value in zip(counts, avg_values):
            if avg_value is None or count <= 0:
                continue
            weighted_sum += float(avg_value) * int(count)
            total_count += int(count)
        timing_summary[key] = {
            "count_total": int(total_count),
            "avg_ms": (None if total_count == 0 else float(weighted_sum / total_count)),
        }
    summary["timing_ms"] = timing_summary
    return summary


def run_single_episode(
    env,
    initial_state,
    task_description: str,
    model: torch.nn.Module,
    action_model: Optional[torch.nn.Module],
    processor: FastWAMProcessor,
    cfg: DictConfig,
    episode_idx: int,
    *,
    action_horizon: int,
    input_w: int,
    input_h: int,
    model_device: str,
    action_device: str,
) -> tuple[bool, list, list[dict[str, Any]], Optional[float], Optional[dict[str, Any]]]:
    if bool(cfg.EVALUATION.get("async_streaming_enabled", False)):
        return run_single_episode_async(
            env=env,
            initial_state=initial_state,
            task_description=task_description,
            video_model=model,
            action_model=(model if action_model is None else action_model),
            processor=processor,
            cfg=cfg,
            episode_idx=episode_idx,
            action_horizon=action_horizon,
            input_w=input_w,
            input_h=input_h,
            video_device=model_device,
        )

    max_steps = _get_max_steps(cfg.EVALUATION.task_suite_name)
    replan_steps = int(cfg.EVALUATION.get("replan_steps", 5))
    num_steps_wait = int(cfg.EVALUATION.get("num_steps_wait", 5))
    use_action_ensembler = bool(cfg.EVALUATION.get("use_action_ensembler", False))
    visualize_future_video = bool(cfg.EVALUATION.get("visualize_future_video", False))
    capture_steps = set(_get_future_frame_capture_steps(cfg)[1:])

    env.reset()
    obs = env.set_init_state(initial_state)
    if use_action_ensembler:
        ensembler = ActionEnsembler()
        ensembler.reset()

    replay_images = []
    predicted_future_video_clips: list[dict[str, Any]] = []
    episode_future_clip_psnr: list[float] = []
    pending_actions: list[list[float]] = []
    current_predicted_future_clip: Optional[dict[str, Any]] = None
    current_replan_step = 0
    current_replan_idx = -1

    t = 0
    done = False
    pbar = tqdm(total=max_steps + num_steps_wait, desc=f"Episode {episode_idx + 1}")
    while t < max_steps + num_steps_wait:
        pbar.update(1)
        if t < num_steps_wait:
            obs, _, done, _ = env.step(get_libero_dummy_action())
            t += 1
            continue

        if len(pending_actions) == 0:
            action_chunk, imgs, predicted_future_frames = _predict_action_chunk(
                obs=obs,
                task_description=task_description,
                model=model,
                processor=processor,
                cfg=cfg,
                action_horizon=action_horizon,
                input_w=input_w,
                input_h=input_h,
                model_device=model_device,
            )
            if predicted_future_frames is not None:
                current_replan_idx += 1
                current_predicted_future_clip = {
                    "replan_idx": current_replan_idx,
                    "gt_frames": [imgs.copy()],
                    "pred_frames": predicted_future_frames,
                }
            else:
                current_predicted_future_clip = None
            current_replan_step = 0
            if use_action_ensembler:
                ensembler.add_actions(action_chunk, t)
                pending_actions = [ensembler.get_action(ts).tolist() for ts in range(t, t + replan_steps)]
            else:
                pending_actions = action_chunk[:replan_steps].tolist()
            replay_images.append(imgs.copy())
        else:
            imgs = get_libero_image(obs)
            replay_images.append(imgs.copy())

        obs, _, done, _ = env.step(pending_actions.pop(0))
        if visualize_future_video and current_predicted_future_clip is not None:
            current_replan_step += 1
            if current_replan_step in capture_steps:
                current_predicted_future_clip["gt_frames"].append(get_libero_image(obs))
            if done or len(pending_actions) == 0:
                expected_frame_count = 1 + sum(
                    1 for capture_step in capture_steps if capture_step <= current_replan_step
                )
                gt_len = len(current_predicted_future_clip["gt_frames"])
                pred_len = len(current_predicted_future_clip["pred_frames"])
                assert gt_len == expected_frame_count, (
                    "GT future frames do not match expected capture count: "
                    f"gt_len={gt_len} expected={expected_frame_count} "
                    f"episode={episode_idx} replan={current_predicted_future_clip['replan_idx']} "
                    f"current_replan_step={current_replan_step} capture_steps={sorted(capture_steps)}."
                )
                assert pred_len >= expected_frame_count, (
                    "Predicted future frames shorter than expected capture count: "
                    f"pred_len={pred_len} expected={expected_frame_count} "
                    f"episode={episode_idx} replan={current_predicted_future_clip['replan_idx']}."
                )
                if pred_len != expected_frame_count:
                    logging.info(
                        "Align predicted clip length to executed steps: "
                        "episode=%s replan=%s done=%s expected=%s pred_full=%s",
                        episode_idx,
                        current_predicted_future_clip["replan_idx"],
                        done,
                        expected_frame_count,
                        pred_len,
                    )
                current_predicted_future_clip["pred_frames"] = current_predicted_future_clip["pred_frames"][
                    :expected_frame_count
                ]
                assert len(current_predicted_future_clip["gt_frames"]) == len(
                    current_predicted_future_clip["pred_frames"]
                ), (
                    "GT/pred frame count mismatch after alignment: "
                    f"len(gt_frames)={len(current_predicted_future_clip['gt_frames'])} "
                    f"len(pred_frames)={len(current_predicted_future_clip['pred_frames'])} "
                    f"episode={episode_idx} replan={current_predicted_future_clip['replan_idx']}."
                )
                clip_psnr = _compute_clip_mean_psnr(
                    current_predicted_future_clip["gt_frames"],
                    current_predicted_future_clip["pred_frames"],
                )
                if clip_psnr is not None:
                    episode_future_clip_psnr.append(clip_psnr)
                predicted_future_video_clips.append(current_predicted_future_clip)
                current_predicted_future_clip = None
        if done:
            break
        t += 1
    pbar.close()

    episode_mean_psnr = (
        float(np.mean(episode_future_clip_psnr)) if len(episode_future_clip_psnr) > 0 else None
    )
    return bool(done), replay_images, predicted_future_video_clips, episode_mean_psnr, None


def run_single_episode_async(
    env,
    initial_state,
    task_description: str,
    video_model: torch.nn.Module,
    action_model: torch.nn.Module,
    processor: FastWAMProcessor,
    cfg: DictConfig,
    episode_idx: int,
    *,
    action_horizon: int,
    input_w: int,
    input_h: int,
    video_device: str,
) -> tuple[bool, list, list[dict[str, Any]], Optional[float], dict[str, Any]]:
    if bool(cfg.EVALUATION.get("visualize_future_video", False)):
        raise ValueError("Async LIBERO rollout does not yet support visualize_future_video=true.")

    if not hasattr(video_model, "start_action_job") or not hasattr(action_model, "start_action_job"):
        raise ValueError("Async LIBERO rollout requires a FastWAMStreaming-style model.")

    max_steps = _get_max_steps(cfg.EVALUATION.task_suite_name)
    num_steps_wait = int(cfg.EVALUATION.get("num_steps_wait", 5))
    obs_stride_env_steps = int(cfg.EVALUATION.get("async_obs_stride_env_steps", 3))
    trigger_every_n_obs = int(cfg.EVALUATION.get("async_action_trigger_every_n_obs", 3))
    video_layers_per_chunk = int(cfg.EVALUATION.get("async_video_layers_per_chunk", 2))
    force_first_job = bool(cfg.EVALUATION.get("async_force_first_job", True))
    control_dt_ms = float(cfg.EVALUATION.get("async_control_dt_ms", 50.0))
    warmup_action_jobs = int(cfg.EVALUATION.get("async_warmup_action_jobs", 8))
    warmup_latest_only = bool(cfg.EVALUATION.get("async_warmup_latest_only", True))
    if warmup_action_jobs < 0:
        raise ValueError(f"`async_warmup_action_jobs` must be >= 0, got {warmup_action_jobs}.")

    prompt = DEFAULT_PROMPT.format(task=task_description)
    with torch.no_grad():
        video_context, video_context_mask = video_model.encode_prompt(prompt)
        if action_model is video_model:
            action_context, action_context_mask = video_context, video_context_mask
        else:
            action_context, action_context_mask = action_model.encode_prompt(prompt)

    action_postprocess = lambda x: _postprocess_libero_action_chunk(x, processor=processor, cfg=cfg)
    runtime = AsyncStreamingActionRuntime(
        video_model=video_model,
        action_model=action_model,
        video_context=video_context,
        video_context_mask=video_context_mask,
        action_context=action_context,
        action_context_mask=action_context_mask,
        action_postprocess=action_postprocess,
        action_horizon=action_horizon,
        num_inference_steps=int(cfg.EVALUATION.get("num_inference_steps", cfg.get("eval_num_inference_steps", 10))),
        sigma_shift=(
            None if cfg.EVALUATION.get("sigma_shift") is None else float(cfg.EVALUATION.get("sigma_shift"))
        ),
        rand_device=str(cfg.EVALUATION.get("rand_device", "cpu")),
        tiled=bool(cfg.EVALUATION.get("tiled", False)),
        action_trigger_every_n_obs=trigger_every_n_obs,
        video_layers_per_chunk=video_layers_per_chunk,
        seed=(None if cfg.get("seed") is None else int(cfg.seed)),
    )

    replay_images = []
    predicted_future_video_clips: list[dict[str, Any]] = []
    episode_future_clip_psnr: list[float] = []
    runtime_started = False
    done = False
    terminated_during_wait = False

    env.reset()
    obs = env.set_init_state(initial_state)
    try:
        runtime.start()
        runtime_started = True
        image, proprio, imgs = _obs_to_model_input(
            obs,
            cfg=cfg,
            processor=processor,
            width=input_w,
            height=input_h,
            device=video_device,
            dtype=video_model.torch_dtype,
        )
        runtime.bootstrap_sync(
            input_image=image,
            obs_index=0,
            obs_timestamp_ms=0.0,
        )

        obs_counter = 0
        warmup_env_step = 0
        warmup_obs_count = 0
        warmup_first_triggered = False
        while runtime.completed_jobs() < warmup_action_jobs:
            if warmup_env_step % obs_stride_env_steps == 0:
                warmup_obs_index = obs_counter
                should_trigger = runtime.should_trigger_on_obs(warmup_obs_count + 1)
                if force_first_job and not warmup_first_triggered:
                    should_trigger = True
                runtime.submit_observation(
                    input_image=image,
                    proprio=proprio,
                    env_step=warmup_env_step,
                    obs_index=warmup_obs_index,
                    obs_timestamp_ms=float(warmup_obs_index) * control_dt_ms * float(obs_stride_env_steps),
                    trigger_job=should_trigger,
                    latest_only_job=bool(should_trigger and warmup_latest_only),
                )
                obs_counter += 1
                warmup_obs_count += 1
                if should_trigger:
                    warmup_first_triggered = True
            obs, _, done, _ = env.step(get_libero_dummy_action())
            if done:
                runtime.wait_until_idle()
                terminated_during_wait = True
                break
            warmup_env_step += 1
            image, proprio, imgs = _obs_to_model_input(
                obs,
                cfg=cfg,
                processor=processor,
                width=input_w,
                height=input_h,
                device=video_device,
                dtype=video_model.torch_dtype,
            )
        if not terminated_during_wait:
            formal_start_step = int(warmup_env_step)

            t = formal_start_step
            first_formal_triggered = False
            pbar = tqdm(total=max_steps, desc=f"Episode {episode_idx + 1} (async)")
            while t < max_steps + formal_start_step:
                pbar.update(1)
                if (t - formal_start_step) % obs_stride_env_steps == 0:
                    formal_obs_index = obs_counter
                    should_trigger = runtime.should_trigger_on_obs()
                    if force_first_job and warmup_action_jobs <= 0 and not first_formal_triggered:
                        should_trigger = True
                    runtime.submit_observation(
                        input_image=image,
                        proprio=proprio,
                        env_step=t,
                        obs_index=formal_obs_index,
                        obs_timestamp_ms=float(formal_obs_index) * control_dt_ms * float(obs_stride_env_steps),
                        trigger_job=should_trigger,
                    )
                    obs_counter += 1
                    if should_trigger:
                        first_formal_triggered = True

                replay_images.append(imgs.copy())
                action = runtime.get_action(t)
                if action is None:
                    action = np.asarray(get_libero_dummy_action(), dtype=np.float32)

                obs, _, done, _ = env.step(action.tolist())
                if done:
                    break
                t += 1
                image, proprio, imgs = _obs_to_model_input(
                    obs,
                    cfg=cfg,
                    processor=processor,
                    width=input_w,
                    height=input_h,
                    device=video_device,
                    dtype=video_model.torch_dtype,
                )
            pbar.close()
    finally:
        if runtime_started:
            runtime.stop()

    runtime_summary = runtime.stats()
    logging.info(
        "Async runtime stats | episode=%s submitted_obs=%s submitted_jobs=%s completed_jobs=%s "
        "actions_served=%s actions_missed=%s dropped_prefix_actions=%s",
        episode_idx,
        runtime_summary["submitted_obs"],
        runtime_summary["submitted_jobs"],
        runtime_summary["completed_jobs"],
        runtime_summary["actions_served"],
        runtime_summary["actions_missed"],
        runtime_summary["dropped_prefix_actions"],
    )
    return (
        bool(done),
        replay_images,
        predicted_future_video_clips,
        float(np.mean(episode_future_clip_psnr)) if len(episode_future_clip_psnr) > 0 else None,
        runtime_summary,
    )


def run_single_task(
    task,
    initial_states,
    model: torch.nn.Module,
    action_model: Optional[torch.nn.Module],
    processor: FastWAMProcessor,
    cfg: DictConfig,
    video_dir: Path,
    predicted_video_dir: Path,
    *,
    action_horizon: int,
    input_w: int,
    input_h: int,
    model_device: str,
    action_device: str,
    render_gpu_device_id: int,
) -> dict:
    env, task_description = get_libero_env(
        task,
        LIBERO_ENV_RESOLUTION,
        cfg.get("seed"),
        render_gpu_device_id=render_gpu_device_id,
    )
    visualize_future_video = bool(cfg.EVALUATION.get("visualize_future_video", False))
    results = {
        "successes": 0,
        "failure_episodes": [],
        "success_episodes": [],
        "task_description": task_description,
        "action_horizon": int(action_horizon),
    }
    async_streaming_enabled = bool(cfg.EVALUATION.get("async_streaming_enabled", False))
    if async_streaming_enabled:
        results["async_runtime_episodes"] = []
        results["async_runtime_summary"] = None
        results["async_video_device"] = str(model_device)
        results["async_action_device"] = str(action_device)
    if visualize_future_video:
        results["episode_future_video_psnr"] = []
        results["future_video_psnr_mean"] = None

    for trial_idx in range(int(cfg.EVALUATION.num_trials)):
        success, replay_images, predicted_future_video_clips, episode_mean_psnr, runtime_summary = run_single_episode(
            env=env,
            initial_state=initial_states[trial_idx],
            task_description=task_description,
            model=model,
            action_model=action_model,
            processor=processor,
            cfg=cfg,
            episode_idx=trial_idx,
            action_horizon=action_horizon,
            input_w=input_w,
            input_h=input_h,
            model_device=model_device,
            action_device=action_device,
        )
        if success:
            results["successes"] += 1
            results["success_episodes"].append(trial_idx)
        else:
            results["failure_episodes"].append(trial_idx)
        if async_streaming_enabled and runtime_summary is not None:
            results["async_runtime_episodes"].append(
                {
                    "episode_idx": int(trial_idx),
                    **runtime_summary,
                }
            )
        if visualize_future_video:
            results["episode_future_video_psnr"].append(episode_mean_psnr)

        save_rollout_video(
            video_dir,
            replay_images,
            f"task{cfg.EVALUATION.task_id}_trial{trial_idx}",
            success=success,
            task_description=task_description,
        )
        if visualize_future_video:
            if len(predicted_future_video_clips) == 0:
                logging.warning(
                    "No predicted future frames collected for task %s trial %s.",
                    cfg.EVALUATION.task_id,
                    trial_idx,
                )
            else:
                all_gt_frames = []
                all_pred_frames = []
                for clip in predicted_future_video_clips:
                    all_gt_frames.extend(clip["gt_frames"])
                    all_pred_frames.extend(clip["pred_frames"])
                    save_prediction_video(
                        predicted_video_dir,
                        clip["gt_frames"],
                        clip["pred_frames"],
                        f"task{cfg.EVALUATION.task_id}_trial{trial_idx}",
                        clip["replan_idx"],
                        success=success,
                        task_description=task_description,
                    )
                save_prediction_video(
                    predicted_video_dir,
                    all_gt_frames,
                    all_pred_frames,
                    f"task{cfg.EVALUATION.task_id}_trial{trial_idx}",
                    "all",
                    success=success,
                    task_description=task_description,
                )

    if visualize_future_video:
        valid_episode_psnr = [x for x in results["episode_future_video_psnr"] if x is not None]
        if len(valid_episode_psnr) > 0:
            results["future_video_psnr_mean"] = float(np.mean(valid_episode_psnr))
    if async_streaming_enabled:
        results["async_runtime_summary"] = _summarize_async_runtime_episodes(results["async_runtime_episodes"])
    return results


@hydra.main(version_base="1.3", config_path="../../configs", config_name="sim_libero.yaml")
def eval_single_process(cfg: DictConfig):
    start_time = time.time()
    partial_state = PartialState()
    partial_state.config = cfg
    render_gpu_device_id = _configure_egl_device(cfg)

    if cfg.get("seed") is not None:
        set_global_seed(int(cfg.seed), get_worker_init_fn=False)

    _validate_visualize_future_video_cfg(cfg)

    env_num = int(cfg.EVALUATION.get("env_num", 1))
    if env_num != 1:
        raise ValueError(
            "Only env_num=1 is supported in eval_libero_single.py. "
            "Use run_libero_manager/run_libero_parallel_test.sh for multi-GPU task parallelism."
        )

    model_device = _resolve_eval_device(cfg)
    model_dtype = _mixed_precision_to_model_dtype(cfg.get("mixed_precision", "bf16"))
    async_video_device, async_action_device = _resolve_async_runtime_devices(cfg, model_device)
    action_model: Optional[torch.nn.Module] = None
    if bool(cfg.EVALUATION.get("async_streaming_enabled", False)) and async_action_device != async_video_device:
        logging.info(
            "Async dual-device runtime enabled: video_device=%s action_device=%s",
            async_video_device,
            async_action_device,
        )
        model = _build_eval_model(cfg, model_dtype=model_dtype, device=async_video_device)
        action_model = _build_eval_model(cfg, model_dtype=model_dtype, device=async_action_device)
    elif bool(cfg.EVALUATION.get("async_streaming_enabled", False)):
        model = _build_eval_model(cfg, model_dtype=model_dtype, device=async_video_device)
    else:
        model = _build_eval_model(cfg, model_dtype=model_dtype, device=model_device)

    dataset_stats_path = _resolve_dataset_stats_path(cfg)
    dataset_stats = load_dataset_stats_from_json(str(dataset_stats_path))
    processor: FastWAMProcessor = instantiate(cfg.data.train.processor).eval()
    processor.set_normalizer_from_stats(dataset_stats)
    logging.info("Using dataset stats: %s", dataset_stats_path)

    action_horizon_cfg = cfg.EVALUATION.get("action_horizon", None)
    if action_horizon_cfg is None:
        action_horizon = int(cfg.data.train.num_frames) - 1
    else:
        action_horizon = int(action_horizon_cfg)
    if action_horizon <= 0:
        raise ValueError(f"EVALUATION.action_horizon must be positive, got {action_horizon}")

    video_size = cfg.data.train.get("video_size", [224, 224])
    if len(video_size) != 2:
        raise ValueError(f"data.train.video_size must be [H, W], got {video_size}")
    input_h = int(video_size[0])
    input_w = int(video_size[1])
    concat_multi_camera = cfg.data.train.get("concat_multi_camera", None)
    shape_meta_images = [meta["shape"] for meta in processor.shape_meta["images"]]

    local_log_dir = Path(cfg.EVALUATION.output_dir)
    local_log_dir.mkdir(parents=True, exist_ok=True)
    video_dir = local_log_dir / cfg.EVALUATION.task_suite_name / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    predicted_video_dir = local_log_dir / cfg.EVALUATION.task_suite_name / "predicted_videos"
    if bool(cfg.EVALUATION.get("visualize_future_video", False)):
        predicted_video_dir.mkdir(parents=True, exist_ok=True)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.EVALUATION.task_suite_name]()
    task = task_suite.get_task(cfg.EVALUATION.task_id)
    initial_states = task_suite.get_task_init_states(cfg.EVALUATION.task_id)

    while len(initial_states) < int(cfg.EVALUATION.num_trials):
        initial_states.extend(initial_states[: (int(cfg.EVALUATION.num_trials) - len(initial_states))])

    results = {
        "task_suite": cfg.EVALUATION.task_suite_name,
        "task_id": cfg.EVALUATION.task_id,
        "task_description": None,
        "action_horizon": int(action_horizon),
        "async_streaming_enabled": bool(cfg.EVALUATION.get("async_streaming_enabled", False)),
        "ckpt_loaded": bool(cfg.get("ckpt") is not None),
        "successes": 0,
        "total_episodes": int(cfg.EVALUATION.num_trials),
        "gpu_id": int(cfg.gpu_id),
        "success_episodes": [],
        "failure_episodes": [],
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "duration": 0,
    }

    logging.info("Running LIBERO evaluation with env_num=1")
    task_results = run_single_task(
        task=task,
        initial_states=initial_states,
        model=model,
        processor=processor,
        cfg=cfg,
        video_dir=video_dir,
        predicted_video_dir=predicted_video_dir,
        action_horizon=action_horizon,
        input_w=input_w,
        input_h=input_h,
        model_device=(async_video_device if bool(cfg.EVALUATION.get("async_streaming_enabled", False)) else model_device),
        action_device=(async_action_device if bool(cfg.EVALUATION.get("async_streaming_enabled", False)) else model_device),
        action_model=action_model,
        render_gpu_device_id=render_gpu_device_id,
    )
    results.update(task_results)

    results["duration"] = time.time() - start_time
    output_dir = Path(cfg.EVALUATION.output_dir) / cfg.EVALUATION.task_suite_name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"gpu{cfg.gpu_id}_task{cfg.EVALUATION.task_id}_results.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, cls=NumpyEncoder)

    print(
        f"Task {cfg.EVALUATION.task_id} completed: "
        f"{results['successes']}/{cfg.EVALUATION.num_trials} successes"
    )
    if results.get("future_video_psnr_mean") is not None:
        print(f"Task {cfg.EVALUATION.task_id} future-video PSNR mean: {results['future_video_psnr_mean']:.4f}")
    print(f"Time taken: {results['duration']:.2f} seconds")
    return results


if __name__ == "__main__":
    eval_single_process()
