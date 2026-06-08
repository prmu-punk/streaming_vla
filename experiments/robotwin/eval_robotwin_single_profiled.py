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


# Importing deploy_policy registers the runtime-backed RobotWin policy on the
# import path `experiments.robotwin.fastwam_policy.deploy_policy`, which is the
# `policy_name` we will feed into RoboTwin's `eval_policy.main()`.
from experiments.robotwin.fastwam_policy import deploy_policy as streaming_policy  # noqa: E402

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


def _compact_episode_stats(episodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    compacted: list[dict[str, Any]] = []
    for ep in episodes:
        item: dict[str, Any] = {}
        if "completed_steps" in ep:
            item["completed_steps"] = ep["completed_steps"]
        if "timing_ms" in ep:
            item["timing_ms"] = {
                key: value
                for key, value in ep["timing_ms"].items()
                if key in {"video_refresh", "action_step", "take_action"}
            }
        if "take_action_internal_steps" in ep:
            item["take_action_internal_steps"] = ep["take_action_internal_steps"]
        if "sampled_denoise_traces" in ep:
            item["sampled_denoise_traces"] = ep["sampled_denoise_traces"]
        compacted.append(item)
    return compacted


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
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--async-obs-stride-env-steps", type=int, default=1)
    # --- model / inference knobs
    p.add_argument("--mixed-precision", default="bf16")
    p.add_argument("--replan-steps", type=int, default=24)
    p.add_argument("--num-inference-steps", type=int, default=16)
    p.add_argument("--action-horizon", type=int, default=8)
    p.add_argument("--sigma-shift", type=float, default=None)
    p.add_argument("--text-cfg-scale", type=float, default=1.0)
    p.add_argument("--negative-prompt", default="")
    p.add_argument("--rand-device", default=None)
    p.add_argument("--tiled", type=int, default=0)
    p.add_argument("--timing-enabled", type=int, default=1)
    p.add_argument("--save-full-runtime-trace", type=int, default=0)
    p.add_argument("--load-text-encoder", type=int, default=0,
                   help="Must be 1 for streaming RobotWin profiling.")
    p.add_argument("--redirect-common-files", type=int, default=1)
    # --- episode loop
    p.add_argument("--eval-num-episodes", type=int, default=1)
    p.add_argument("--instruction-type", default="seen")
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
    rand_device = args.rand_device
    if rand_device is None:
        rand_device = args.device
    return {
        # RoboTwin main() fields
        "task_name": args.task_name,
        "task_config": args.task_config,
        "ckpt_setting": str(Path(args.ckpt_setting).resolve()),
        "policy_name": "experiments.robotwin.fastwam_policy.deploy_policy",
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
        "rand_device": rand_device,
        "tiled": bool(args.tiled),
        "timing_enabled": bool(args.timing_enabled),
        "profile_runtime": True,
        "save_full_runtime_trace": bool(args.save_full_runtime_trace),
        "load_text_encoder": bool(args.load_text_encoder),
        "redirect_common_files": bool(args.redirect_common_files),
        "device": args.device,
        "async_obs_stride_env_steps": args.async_obs_stride_env_steps,
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
        compact_episodes = _compact_episode_stats(episodes)

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
            "device": str(args.device),
            "async_runtime_episodes": compact_episodes,
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
