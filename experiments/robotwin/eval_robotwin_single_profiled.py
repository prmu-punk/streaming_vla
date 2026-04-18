"""
RoboTwin streaming-pipeline profiling entrypoint — REAL machine (full SAPIEN).

This delegates the entire episode loop to RoboTwin's upstream `script/eval_policy.py:main()` 
so we inherit every piece of real-env behaviour (asset loading, task_config yaml, instruction sampler,
seeds, eval_video ffmpeg pipe, cache clearing). After `main()` returns we pull
the built `StreamingWorldActionRobotWinPolicy` via `get_last_policy()` and dump
a JSON in the same schema as libero profiling.

Usage (expects to be run from the Streaming_VLA repo root):

    bash experiments/robotwin/run_profile.sh click_alarmclock demo_randomized

or equivalently:

    PYTHONPATH=src:. .venv/bin/python \
        experiments/robotwin/eval_robotwin_single_profiled.py \
        --task-name click_alarmclock \
        --task-config demo_randomized \
        --ckpt-setting checkpoints/fastwam_release/robotwin_uncond_3cam_384.pt \
        --dataset-stats checkpoints/fastwam_release/robotwin_uncond_3cam_384_dataset_stats.json \
        --eval-num-episodes 1 \
        --async-video-device cuda:0 \
        --async-action-device cuda:1 \
        --profile-output-dir ./evaluate_results/robotwin_profile/real

Output:
    <profile-output-dir>/<task_name>/gpu0_task<task_name>_results.json
        (schema matches libero profiling:
         async_runtime_episodes / async_runtime_summary / layer_source_stats)
"""
from __future__ import annotations

import argparse
import faulthandler
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
ROBOTWIN_ROOT = PROJECT_ROOT / "third_party" / "RoboTwin"
ROBOTWIN_SCRIPT_DIR = ROBOTWIN_ROOT / "script"
ROBOTWIN_POLICY_DIR = ROBOTWIN_ROOT / "policy"

for p in (PROJECT_ROOT, SRC_ROOT, ROBOTWIN_ROOT, ROBOTWIN_SCRIPT_DIR, ROBOTWIN_POLICY_DIR):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)


# Importing deploy_policy registers the streaming policy on the import path
# `experiments.robotwin.fastwam_streaming_policy.deploy_policy`, which is the
# `policy_name` we will feed into RoboTwin's `eval_policy.main()`.
from experiments.robotwin.fastwam_streaming_policy import deploy_policy as streaming_policy  # noqa: E402

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
logger = logging.getLogger(__name__)


class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def _summarize_episodes(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    if not episodes:
        return {"num_episodes": 0}
    summary: dict[str, Any] = {"num_episodes": len(episodes)}
    scalar_keys = [
        "submitted_obs",
        "submitted_jobs",
        "completed_jobs",
        "actions_served",
        "actions_missed",
        "dropped_prefix_actions",
    ]
    for key in scalar_keys:
        values = [float(ep.get(key, 0)) for ep in episodes]
        summary[f"{key}_total"] = float(np.sum(values))
        summary[f"{key}_mean"] = float(np.mean(values))
    timing_keys = ["video_refresh", "action_job", "action_job_wall", "action_step", "snapshot_copy", "env_step"]
    timing_summary: dict[str, Any] = {}
    for key in timing_keys:
        weighted = 0.0
        total = 0
        for ep in episodes:
            entry = ep.get("timing_ms", {}).get(key, {}) or {}
            c = int(entry.get("count", 0))
            a = entry.get("avg_ms")
            if a is None or c <= 0:
                continue
            weighted += float(a) * c
            total += c
        timing_summary[key] = {
            "count_total": int(total),
            "avg_ms": (None if total == 0 else float(weighted / total)),
        }
    summary["timing_ms"] = timing_summary
    return summary


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    # --- required
    p.add_argument("--task-name", required=True, help="RoboTwin task (e.g. click_alarmclock).")
    p.add_argument("--task-config", default="demo_randomized",
                   help="RoboTwin task_config yaml stem (demo_clean | demo_randomized).")
    p.add_argument("--ckpt-setting", required=True,
                   help="Path to the FastWAM streaming checkpoint (.pt).")
    p.add_argument("--dataset-stats", required=True,
                   help="Path to dataset_stats.json used by the FastWAM processor.")
    # --- streaming runtime knobs
    p.add_argument("--async-video-device", default="cuda:0")
    p.add_argument("--async-action-device", default="cuda:1")
    p.add_argument("--async-obs-stride-env-steps", type=int, default=3)
    p.add_argument("--async-action-trigger-every-n-obs", type=int, default=3)
    p.add_argument("--async-video-layers-per-chunk", type=int, default=2)
    p.add_argument("--async-force-first-job", type=int, default=0)
    p.add_argument("--async-warmup-action-jobs", type=int, default=20)
    p.add_argument("--async-control-dt-ms", type=float, default=150.0)
    # --- model / inference knobs
    p.add_argument("--mixed-precision", default="bf16")
    p.add_argument("--replan-steps", type=int, default=24)
    p.add_argument("--num-inference-steps", type=int, default=8)
    p.add_argument("--action-horizon", type=int, default=None)
    p.add_argument("--sigma-shift", type=float, default=None)
    p.add_argument("--text-cfg-scale", type=float, default=1.0)
    p.add_argument("--negative-prompt", default="")
    p.add_argument("--rand-device", default="cpu")
    p.add_argument("--tiled", type=int, default=0)
    p.add_argument("--timing-enabled", type=int, default=1)
    p.add_argument("--load-text-encoder", type=int, default=1,
                   help="Must be 1 for streaming RobotWin profiling.")
    p.add_argument("--redirect-common-files", type=int, default=1)
    # --- episode loop
    p.add_argument("--eval-num-episodes", type=int, default=1)
    p.add_argument("--instruction-type", default="unseen")
    p.add_argument("--skip-get-obs-within-replan", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-output-dir", default=None,
                   help="RoboTwin's own eval_result output dir (success-rate txt). "
                        "Defaults to RoboTwin's built-in path if unset.")
    # --- our profile JSON
    p.add_argument("--profile-output-dir", default="./evaluate_results/robotwin_profile/real")
    p.add_argument("--gpu-tag", default="0")
    return p.parse_args()


def _build_usr_args(args: argparse.Namespace) -> Dict[str, Any]:
    sim_cfg_path = str((PROJECT_ROOT / "configs" / "sim_robotwin.yaml").resolve())
    return {
        # RoboTwin main() fields
        "task_name": args.task_name,
        "task_config": args.task_config,
        "ckpt_setting": str(Path(args.ckpt_setting).resolve()),
        "policy_name": "experiments.robotwin.fastwam_streaming_policy.deploy_policy",
        "instruction_type": args.instruction_type,
        "skip_get_obs_within_replan": bool(args.skip_get_obs_within_replan),
        "eval_num_episodes": int(args.eval_num_episodes),
        "eval_output_dir": args.eval_output_dir,
        "seed": int(args.seed),
        # StreamingWorldActionRobotWinPolicy fields
        "sim_cfg_path": sim_cfg_path,
        "sim_task": "robotwin_streaming_action_ft_3cam_384_1e-4",
        "mixed_precision": args.mixed_precision,
        "dataset_stats_path": str(Path(args.dataset_stats).resolve()),
        "action_horizon": args.action_horizon,
        "replan_steps": args.replan_steps,
        "num_inference_steps": args.num_inference_steps,
        "sigma_shift": args.sigma_shift,
        "text_cfg_scale": args.text_cfg_scale,
        "negative_prompt": args.negative_prompt,
        "rand_device": args.rand_device,
        "tiled": bool(args.tiled),
        "timing_enabled": bool(args.timing_enabled),
        "load_text_encoder": bool(args.load_text_encoder),
        "redirect_common_files": bool(args.redirect_common_files),
        "async_video_device": args.async_video_device,
        "async_action_device": args.async_action_device,
        "async_obs_stride_env_steps": args.async_obs_stride_env_steps,
        "async_action_trigger_every_n_obs": args.async_action_trigger_every_n_obs,
        "async_video_layers_per_chunk": args.async_video_layers_per_chunk,
        "async_force_first_job": bool(args.async_force_first_job),
        "async_warmup_action_jobs": args.async_warmup_action_jobs,
        "async_control_dt_ms": args.async_control_dt_ms,
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = _parse_args()
    usr_args = _build_usr_args(args)
    os.environ.setdefault("PYTHONFAULTHANDLER", "1")
    preferred_tmp = os.environ.get("FASTWAM_TMPDIR", "/tmp/fw")
    tmp_root = Path(preferred_tmp)
    # torch_shm_manager uses UNIX sockets; overly long TMPDIR can cause EINVAL.
    if len(str(tmp_root)) > 60:
        fallback_tmp = Path("/tmp/fw")
        logger.warning(
            "FASTWAM_TMPDIR is too long for safe torch_shm_manager socket paths (%s). "
            "Falling back to %s.",
            tmp_root,
            fallback_tmp,
        )
        tmp_root = fallback_tmp
    tmp_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TMPDIR", str(tmp_root))
    os.environ.setdefault("TMP", str(tmp_root))
    os.environ.setdefault("TEMP", str(tmp_root))
    try:
        import torch.multiprocessing as torch_mp

        torch_mp.set_sharing_strategy("file_descriptor")
        logger.info(
            "Set torch multiprocessing sharing strategy to file_descriptor (TMPDIR=%s).",
            tmp_root,
        )
    except Exception:
        logger.exception("Failed to set torch multiprocessing sharing strategy file_descriptor.")

    fault_dir = Path(args.profile_output_dir).resolve() / args.task_name
    fault_dir.mkdir(parents=True, exist_ok=True)
    os.environ["FASTWAM_FAULT_DIR"] = str(fault_dir)
    os.environ.setdefault("FASTWAM_CHILD_FAULTHANDLER", "1")
    fault_log_file = fault_dir / f"gpu{args.gpu_tag}_task{args.task_name}_faulthandler.log"
    fault_log_handle = open(fault_log_file, "a", encoding="utf-8")
    try:
        faulthandler.enable(file=fault_log_handle, all_threads=True)
    except Exception:
        logger.exception("Failed to enable faulthandler file logging; falling back to stderr.")
        faulthandler.enable(all_threads=True)

    try:
        # RoboTwin's main() reads `./task_config/<task_config>.yml` relative to
        # cwd, so chdir into the RoboTwin root for the duration of the run.
        original_cwd = Path.cwd()
        os.chdir(ROBOTWIN_ROOT)
        logger.info("Changed cwd to %s for RoboTwin main()", ROBOTWIN_ROOT)

        start_time = time.time()
        try:
            # Import lazily so all the `sys.path.insert`s + chdir are in effect.
            from script import eval_policy as robotwin_eval_policy  # type: ignore

            logger.info("Delegating to RoboTwin eval_policy.main with usr_args=%s", usr_args)
            robotwin_eval_policy.main(usr_args)
        except BaseException:
            logger.exception("RoboTwin eval_policy.main failed; attempting streaming policy teardown.")
            try:
                policy = streaming_policy.get_last_policy()
                if policy is not None:
                    policy.collect_episode_stats()
            except Exception:
                logger.exception("Failed during emergency streaming policy teardown.")
            raise
        finally:
            os.chdir(original_cwd)

        duration = time.time() - start_time

        policy = streaming_policy.get_last_policy()
        if policy is None:
            raise RuntimeError(
                "StreamingWorldActionRobotWinPolicy was never built — did RoboTwin "
                "main() fail before calling get_model?"
            )
        episodes = policy.collect_episode_stats()

        task_name = args.task_name
        output_root = Path(args.profile_output_dir)
        output_dir = output_root / task_name
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"gpu{args.gpu_tag}_task{task_name}_results.json"

        payload = {
            "task_suite": "robotwin_real",
            "task_id": task_name,
            "task_config": args.task_config,
            "instruction_type": args.instruction_type,
            "ckpt_loaded": True,
            "total_episodes": int(args.eval_num_episodes),
            "duration": float(duration),
            "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "async_video_device": str(args.async_video_device),
            "async_action_device": str(args.async_action_device),
            "async_runtime_episodes": episodes,
            "async_runtime_summary": _summarize_episodes(episodes),
            "layer_source_stats": episodes[-1].get("layer_source_stats") if episodes else None,
        }

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=4, cls=_NumpyEncoder)

        logger.info("Wrote profile results to %s", output_file)
        logger.info("Total duration: %.2f s across %d episodes", duration, args.eval_num_episodes)
    finally:
        try:
            fault_log_handle.flush()
        except Exception:
            pass
        try:
            os.fsync(fault_log_handle.fileno())
        except Exception:
            pass
        fault_log_handle.close()


if __name__ == "__main__":
    main()
