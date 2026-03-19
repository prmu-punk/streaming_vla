from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys
import os

# 添加项目根目录到 sys.path 以解决找不到 model.qwen3_vl 的问题
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))


def main() -> None:
    """训练启动器入口，转发参数到 `train_libero90_async.py`。

    接口对应:
    - 输入接口: 命令行参数 `--config/--run-name/--extra`。
    - 输出接口: 以子进程方式执行 Hydra 训练脚本并沿用其退出码。
    """
    parser = argparse.ArgumentParser(description="Run RTC async training from RTC_Flow")
    parser.add_argument(
        "--config",
        type=str,
        default="RTC_Flow/configs/train_libero90_async.yaml",
        help="Path to RTC_Flow training config (relative to Streaming_VLA root or absolute). Default config points to local libero10_N500 zarr_path.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="rtc_async_run",
        help="Hydra run name under RTC_Flow/outputs/runs.",
    )
    parser.add_argument(
        "--extra",
        type=str,
        nargs="*",
        default=[],
        help="Extra Hydra overrides, e.g. dataset.zarr_path=/data/libero.zarr training.num_epochs=10",
    )
    args = parser.parse_args()

    repo_root = pathlib.Path(__file__).resolve().parents[2]
    run_dir = repo_root / "RTC_Flow" / "outputs" / "runs" / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        str(repo_root / "RTC_Flow" / "scripts" / "train_libero90_async.py"),
        f"--config-path={repo_root / 'RTC_Flow' / 'configs'}",
        "--config-name=train_libero90_async",
        f"hydra.run.dir={run_dir}",
    ]
    command.extend(args.extra)

    subprocess.run(command, check=True, cwd=str(repo_root))


if __name__ == "__main__":
    main()
