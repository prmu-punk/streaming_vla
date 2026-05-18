from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.robotwin.eval_dataset_obs_mse_common import NumpyEncoder, run_runtime_mse  # noqa: E402


DEFAULT_TASK_CONFIG = "robotwin_streaming_action_ft_3cam_384_1e-4"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run RoboTwin dataset observations through the async runtime and report served-action MSE."
    )
    parser.add_argument("--task-config", default=DEFAULT_TASK_CONFIG)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--dataset-index", type=int, default=0)
    parser.add_argument("--episode-index", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=160)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default=None)
    parser.add_argument("--mixed-precision", default="bf16")
    parser.add_argument("--num-inference-steps", type=int, default=None)
    parser.add_argument("--action-horizon", type=int, default=None)
    parser.add_argument("--rand-device", default=None)
    parser.add_argument("--control-dt-ms", type=float, default=None)
    parser.add_argument("--obs-stride-env-steps", type=int, default=None)
    parser.add_argument("--skip-initial-steps", type=int, default=0)
    parser.add_argument("--no-realtime-pacing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_runtime_mse(args)
    output_path = Path(args.output_json).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, cls=NumpyEncoder)
    print(f"Wrote RoboTwin dataset-observation runtime MSE to: {output_path}")
    print(json.dumps(result["summary"], indent=2, cls=NumpyEncoder))


if __name__ == "__main__":
    main()
