from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys
def main() -> None:
    """在线评估启动器入口，转发参数到 `eval_libero90_rtc_online.py`。

    接口对应:
    - 输入接口: 命令行参数（checkpoint、task、调度参数等）。
    - 输出接口: 子进程执行评估脚本；成功时产出评估日志/可选视频。
    """
    parser = argparse.ArgumentParser(description="Run RTC online eval")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained action_expert checkpoint.")
    parser.add_argument("--task", type=str, required=True, help="LIBERO task name.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train_libero90_async.yaml",
        help="Path to training config used for eval settings.",
    )
    parser.add_argument("--match-rank", type=int, default=0)
    parser.add_argument("--num-frames", type=int, default=6)
    parser.add_argument("--max-control-cycles", type=int, default=120)
    parser.add_argument("--inference-delay", type=int, default=None)
    parser.add_argument("--execute-horizon", type=int, default=None)
    parser.add_argument("--save-video", action="store_true")
    args = parser.parse_args()

    repo_root = pathlib.Path(__file__).resolve().parents[1]
    command = [
        sys.executable,
        str(repo_root / "scripts" / "eval_libero90_rtc_online.py"),
        "--checkpoint",
        args.checkpoint,
        "--task",
        args.task,
        "--config",
        args.config,
        "--match-rank",
        str(args.match_rank),
        "--num-frames",
        str(args.num_frames),
        "--max-control-cycles",
        str(args.max_control_cycles),
    ]
    if args.inference_delay is not None and args.execute_horizon is not None:
        command.extend(["--inference-delay", str(args.inference_delay), "--execute-horizon", str(args.execute_horizon)])
    if args.save_video:
        command.append("--save-video")

    subprocess.run(command, check=True, cwd=str(repo_root))


if __name__ == "__main__":
    main()
