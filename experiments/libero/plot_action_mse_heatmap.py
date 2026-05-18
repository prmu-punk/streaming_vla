from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SQERR_OUTLIER_THRESHOLD = 1.0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot per-action per-dimension squared error heatmaps from sync and async "
            "dataset-observation MSE result JSONs."
        )
    )
    parser.add_argument("--sync-json", required=True)
    parser.add_argument("--async-json", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument(
        "--same-color-scale",
        action="store_true",
        help="Use the same vmax for both heatmaps.",
    )
    return parser.parse_args()


def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _sync_sqerr_matrix(data: dict) -> np.ndarray:
    rows = data.get("per_step", [])
    per_step = []
    for row in rows:
        pred = np.asarray(row["first_pred_action"], dtype=np.float32)
        target = np.asarray(row["first_target_action"], dtype=np.float32)
        per_step.append(np.square(pred - target))
    if not per_step:
        raise ValueError("Sync JSON has no per_step data.")
    return np.stack(per_step, axis=0)


def _async_sqerr_matrix(data: dict) -> np.ndarray:
    rows = data.get("per_step", [])
    per_step = [np.asarray(row["per_dim_sqerr"], dtype=np.float32) for row in rows]
    if not per_step:
        raise ValueError("Async JSON has no per_step data.")
    return np.stack(per_step, axis=0)


def _truncate_pair(sync_sqerr: np.ndarray, async_sqerr: np.ndarray, max_steps: int | None) -> tuple[np.ndarray, np.ndarray]:
    common_steps = min(int(sync_sqerr.shape[0]), int(async_sqerr.shape[0]))
    if max_steps is not None:
        common_steps = min(common_steps, int(max_steps))
    if common_steps <= 0:
        raise ValueError("No overlapping steps to plot.")
    return sync_sqerr[:common_steps], async_sqerr[:common_steps]


def _plot_heatmaps(sync_sqerr: np.ndarray, async_sqerr: np.ndarray, output_path: Path, same_color_scale: bool) -> None:
    if int(sync_sqerr.shape[1]) != int(async_sqerr.shape[1]):
        raise ValueError(
            f"Dimension mismatch between sync ({sync_sqerr.shape}) and async ({async_sqerr.shape})."
        )

    dims = int(sync_sqerr.shape[1])
    steps = int(sync_sqerr.shape[0])
    sync_img = np.ma.masked_greater(sync_sqerr.T, SQERR_OUTLIER_THRESHOLD)
    async_img = np.ma.masked_greater(async_sqerr.T, SQERR_OUTLIER_THRESHOLD)

    if same_color_scale:
        vmax = float(max(np.ma.max(sync_img), np.ma.max(async_img)))
        sync_vmax = vmax
        async_vmax = vmax
    else:
        sync_vmax = float(np.ma.max(sync_img))
        async_vmax = float(np.ma.max(async_img))

    fig, axes = plt.subplots(1, 2, figsize=(max(10, steps * 0.35), 5), constrained_layout=True)
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad(color="black")

    im0 = axes[0].imshow(sync_img, aspect="auto", origin="lower", cmap=cmap, vmin=0.0, vmax=sync_vmax)
    axes[0].set_title("Sync per-dim sqerr")
    axes[0].set_xlabel("env_step")
    axes[0].set_ylabel("action dim")
    axes[0].set_yticks(range(dims))
    axes[0].set_xticks(range(0, steps, max(1, steps // 8)))
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(async_img, aspect="auto", origin="lower", cmap=cmap, vmin=0.0, vmax=async_vmax)
    axes[1].set_title("Async per-dim sqerr")
    axes[1].set_xlabel("env_step")
    axes[1].set_ylabel("action dim")
    axes[1].set_yticks(range(dims))
    axes[1].set_xticks(range(0, steps, max(1, steps // 8)))
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    sync_data = _load_json(args.sync_json)
    async_data = _load_json(args.async_json)
    sync_sqerr = _sync_sqerr_matrix(sync_data)
    async_sqerr = _async_sqerr_matrix(async_data)
    sync_sqerr, async_sqerr = _truncate_pair(sync_sqerr, async_sqerr, args.max_steps)
    _plot_heatmaps(
        sync_sqerr=sync_sqerr,
        async_sqerr=async_sqerr,
        output_path=Path(args.output).resolve(),
        same_color_scale=bool(args.same_color_scale),
    )
    print(f"Saved heatmap to {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
