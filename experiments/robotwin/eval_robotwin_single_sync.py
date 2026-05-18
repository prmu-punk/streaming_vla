"""
RobotWin single-task synchronous FastWAM evaluation entrypoint.

This runs the restored chunk-replan policy in
`experiments/robotwin/fastwam_policy_sync` and uses RoboTwin's native eval
loop, including its eval-video writer when the task config enables it.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
POLICY_NAME = "fastwam_policy_sync"


def _resolve_path(path_str: str, *, base: Path) -> Path:
    path = Path(os.path.expanduser(os.path.expandvars(str(path_str))))
    if not path.is_absolute():
        path = (base / path).resolve()
    return path.resolve()


def _resolve_optional_path(path_value: Any, *, base: Path) -> Path | None:
    if path_value is None:
        return None
    text = str(path_value).strip()
    if text == "" or text.lower() in {"none", "null"}:
        return None
    return _resolve_path(text, base=base)


def _resolve_dataset_stats_path(dataset_stats: str | None, ckpt_path: Path) -> Path:
    explicit = _resolve_optional_path(dataset_stats, base=PROJECT_ROOT)
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(explicit)
    for parent in list(ckpt_path.parents)[:4]:
        candidates.append((parent / "dataset_stats.json").resolve())

    seen: set[Path] = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.exists():
            return resolved

    raise FileNotFoundError(
        "Failed to locate dataset_stats.json. Pass --dataset-stats explicitly."
    )


def _ensure_policy_symlink(robotwin_root: Path, policy_source_dir: Path) -> Path:
    policy_root = robotwin_root / "policy"
    if not policy_root.is_dir():
        raise FileNotFoundError(f"RoboTwin policy directory not found: {policy_root}")

    policy_target = policy_root / POLICY_NAME
    source_resolved = policy_source_dir.resolve()

    if not policy_target.exists() and not policy_target.is_symlink():
        policy_target.symlink_to(source_resolved, target_is_directory=True)
        return policy_target

    if policy_target.is_symlink():
        target_resolved = policy_target.resolve()
        if target_resolved != source_resolved:
            raise RuntimeError(
                f"Policy symlink conflict: {policy_target} -> {target_resolved}, "
                f"expected -> {source_resolved}"
            )
        return policy_target

    raise RuntimeError(
        f"Path already exists and is not a symlink: {policy_target}. "
        "Please handle it manually to avoid overriding existing policy files."
    )


def _format_override_value(value: Any) -> str:
    if isinstance(value, bool):
        return "True" if value else "False"
    if value is None:
        return "None"
    if isinstance(value, (int, float)):
        return str(value)
    return repr(str(value))


def _append_override(overrides: list[str], key: str, value: Any, *, skip_none: bool = True) -> None:
    if skip_none and value is None:
        return
    overrides.extend([f"--{key}", _format_override_value(value)])


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task-name", default="click_alarmclock")
    p.add_argument("--task-config", default="demo_randomized")
    p.add_argument("--ckpt-setting", required=True)
    p.add_argument("--dataset-stats", default=None)
    p.add_argument("--robotwin-root", default="third_party/RoboTwin")
    p.add_argument("--output-dir", default="./evaluate_results/robotwin_sync/real")
    p.add_argument("--eval-num-episodes", type=int, default=1)
    p.add_argument("--instruction-type", default="unseen")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--gpu-id", default=None)
    p.add_argument("--sim-task", default="robotwin_streaming_action_ft_3cam_384_1e-4")
    p.add_argument("--mixed-precision", default="bf16")
    p.add_argument("--action-horizon", type=int, default=10)
    p.add_argument("--replan-steps", type=int, default=8)
    p.add_argument("--num-inference-steps", type=int, default=10)
    p.add_argument("--sigma-shift", type=float, default=None)
    p.add_argument("--text-cfg-scale", type=float, default=1.0)
    p.add_argument("--negative-prompt", default="")
    p.add_argument("--rand-device", default="cuda")
    p.add_argument("--tiled", type=int, default=0)
    p.add_argument("--timing-enabled", type=int, default=1)
    p.add_argument("--skip-get-obs-within-replan", type=int, default=1)
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    ckpt_path = _resolve_path(args.ckpt_setting, base=PROJECT_ROOT)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    dataset_stats_path = _resolve_dataset_stats_path(args.dataset_stats, ckpt_path)
    robotwin_root = _resolve_path(args.robotwin_root, base=PROJECT_ROOT)
    policy_source_dir = (PROJECT_ROOT / "experiments" / "robotwin" / POLICY_NAME).resolve()
    _ensure_policy_symlink(robotwin_root=robotwin_root, policy_source_dir=policy_source_dir)

    output_root = _resolve_path(args.output_dir, base=PROJECT_ROOT)
    task_output_dir = output_root / args.task_name
    task_output_dir.mkdir(parents=True, exist_ok=True)
    log_file = task_output_dir / f"sync_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    sim_cfg_path = (PROJECT_ROOT / "configs" / "sim_robotwin.yaml").resolve()

    overrides: list[str] = []
    _append_override(overrides, "task_name", args.task_name)
    _append_override(overrides, "task_config", args.task_config)
    _append_override(overrides, "ckpt_setting", str(ckpt_path))
    _append_override(overrides, "seed", args.seed)
    _append_override(overrides, "policy_name", POLICY_NAME)
    _append_override(overrides, "instruction_type", args.instruction_type)
    _append_override(overrides, "eval_num_episodes", args.eval_num_episodes)
    _append_override(overrides, "sim_cfg_path", str(sim_cfg_path))
    _append_override(overrides, "sim_task", args.sim_task)
    _append_override(overrides, "eval_output_dir", str(task_output_dir))
    _append_override(overrides, "mixed_precision", args.mixed_precision)
    _append_override(overrides, "device", args.device)
    _append_override(overrides, "dataset_stats_path", str(dataset_stats_path))
    _append_override(overrides, "action_horizon", args.action_horizon)
    _append_override(overrides, "replan_steps", args.replan_steps)
    _append_override(overrides, "num_inference_steps", args.num_inference_steps)
    _append_override(overrides, "sigma_shift", args.sigma_shift)
    _append_override(overrides, "text_cfg_scale", args.text_cfg_scale)
    _append_override(overrides, "negative_prompt", args.negative_prompt)
    _append_override(overrides, "rand_device", args.rand_device)
    _append_override(overrides, "tiled", bool(args.tiled))
    _append_override(overrides, "timing_enabled", bool(args.timing_enabled))
    _append_override(overrides, "skip_get_obs_within_replan", bool(args.skip_get_obs_within_replan))

    cmd = [
        sys.executable,
        "-u",
        "script/eval_policy.py",
        "--config",
        f"policy/{POLICY_NAME}/deploy_policy.yml",
        "--overrides",
        *overrides,
    ]

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPATH"] = f"{PROJECT_ROOT / 'src'}:{PROJECT_ROOT}{env.get('PYTHONPATH', '') and ':' + env['PYTHONPATH']}"
    if args.gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    print(f"[sync_eval] task       = {args.task_name} ({args.task_config})")
    print(f"[sync_eval] episodes   = {args.eval_num_episodes}")
    print(f"[sync_eval] ckpt       = {ckpt_path}")
    print(f"[sync_eval] stats      = {dataset_stats_path}")
    print(f"[sync_eval] output     = {task_output_dir}")
    print(f"[sync_eval] log        = {log_file}")

    with open(log_file, "w", encoding="utf-8") as log_f:
        process = subprocess.Popen(
            cmd,
            cwd=str(robotwin_root),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log_f.write(line)
            log_f.flush()
        return_code = process.wait()

    if return_code != 0:
        raise RuntimeError(f"RoboTwin sync evaluation failed with return code {return_code}. Log: {log_file}")


if __name__ == "__main__":
    main()
