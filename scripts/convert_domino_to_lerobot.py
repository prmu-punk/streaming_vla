#!/usr/bin/env python3
"""Convert DOMINO/RoboTwin-style HDF5 demonstrations to FastWAM LeRobot data.

The generated dataset matches the local LeRobot v2.1 layout consumed by
``RobotVideoDataset``.  DOMINO stores JPEG bytes in HDF5; this script decodes
them, writes three synchronized MP4 streams, and stores qpos state/action rows
in parquet.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import h5py
import imageio.v2 as imageio
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


CAMERA_MAP = {
    "head_camera": "cam_high",
    "left_camera": "cam_left_wrist",
    "right_camera": "cam_right_wrist",
}

JOINT_NAMES = [
    "left_waist",
    "left_shoulder",
    "left_elbow",
    "left_forearm_roll",
    "left_wrist_angle",
    "left_wrist_rotate",
    "left_gripper",
    "right_waist",
    "right_shoulder",
    "right_elbow",
    "right_forearm_roll",
    "right_wrist_angle",
    "right_wrist_rotate",
    "right_gripper",
]


@dataclass(frozen=True)
class Episode:
    source_path: Path
    source_index: int
    output_index: int
    task: str
    task_index: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        type=Path,
        required=True,
        help="DOMINO task config directory, e.g. DOMINO_fastwam/data/adjust_bottle/demo_clean_dynamic",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Output LeRobot dataset directory consumed by configs/data/domino.yaml",
    )
    parser.add_argument("--fps", type=int, default=50)
    parser.add_argument("--chunks-size", type=int, default=1000)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument(
        "--instruction-split",
        choices=("seen", "unseen"),
        default="seen",
        help="Instruction split to read from DOMINO instructions/episode*.json if present.",
    )
    parser.add_argument("--instruction-index", type=int, default=0)
    parser.add_argument(
        "--default-task",
        type=str,
        default=None,
        help="Fallback instruction when no DOMINO instruction json exists.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove the existing output directory before writing.",
    )
    return parser.parse_args()


def episode_number(path: Path) -> int:
    match = re.search(r"episode(\d+)\.hdf5$", path.name)
    if match is None:
        raise ValueError(f"Cannot parse episode index from {path}")
    return int(match.group(1))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=4, ensure_ascii=False), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def decode_jpeg_bytes(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        raw = value.tobytes()
    else:
        raw = bytes(value)
    raw = raw.rstrip(b"\x00")
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode JPEG bytes from DOMINO HDF5")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def read_camera_frames(h5: h5py.File, camera_name: str) -> np.ndarray:
    dataset = h5[f"observation/{camera_name}/rgb"]
    frames = [decode_jpeg_bytes(dataset[i]) for i in range(len(dataset))]
    return np.stack(frames, axis=0)


def read_instruction(input_root: Path, ep_idx: int, split: str, instruction_index: int, fallback: str) -> str:
    path = input_root / "instructions" / f"episode{ep_idx}.json"
    if not path.exists():
        return fallback

    data = json.loads(path.read_text(encoding="utf-8"))
    candidates = data.get(split)
    if not candidates and split != "seen":
        candidates = data.get("seen")
    if not candidates:
        return fallback

    idx = max(0, min(instruction_index, len(candidates) - 1))
    return str(candidates[idx])


def video_feature(height: int, width: int, fps: int) -> dict[str, Any]:
    return {
        "dtype": "video",
        "shape": [height, width, 3],
        "names": ["height", "width", "rgb"],
        "info": {
            "video.height": height,
            "video.width": width,
            "video.codec": "h264",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "video.fps": fps,
            "video.channels": 3,
            "has_audio": False,
        },
    }


def numeric_feature(dtype: str, shape: list[int], names: Any = None) -> dict[str, Any]:
    return {"dtype": dtype, "shape": shape, "names": names}


def compute_basic_stats(values: np.ndarray) -> dict[str, list[float]]:
    arr = values.astype(np.float64)
    return {
        "min": arr.min(axis=0).tolist(),
        "max": arr.max(axis=0).tolist(),
        "mean": arr.mean(axis=0).tolist(),
        "std": arr.std(axis=0).tolist(),
        "count": [int(arr.shape[0])],
    }


def episode_lerobot_stats(columns: dict[str, np.ndarray]) -> dict[str, Any]:
    stats: dict[str, Any] = {}
    for key, values in columns.items():
        arr = values
        if arr.ndim == 1:
            arr = arr[:, None]
        stats[key] = compute_basic_stats(arr)
    return stats


def fastwam_dataset_stats(states: list[np.ndarray], actions: list[np.ndarray]) -> dict[str, Any]:
    state = np.concatenate(states, axis=0).astype(np.float32)
    action = np.concatenate(actions, axis=0).astype(np.float32)

    def one(arr: np.ndarray) -> dict[str, Any]:
        global_min = arr.min(axis=0)
        global_max = arr.max(axis=0)
        global_mean = arr.mean(axis=0)
        global_std = arr.std(axis=0)
        q01 = np.quantile(arr, 0.01, axis=0)
        q99 = np.quantile(arr, 0.99, axis=0)
        step = lambda x: x[None, :].tolist()
        return {
            "stepwise_min": step(global_min),
            "stepwise_max": step(global_max),
            "global_min": global_min.tolist(),
            "global_max": global_max.tolist(),
            "stepwise_q01": step(q01),
            "stepwise_q99": step(q99),
            "global_q01": q01.tolist(),
            "global_q99": q99.tolist(),
            "stepwise_mean": step(global_mean),
            "stepwise_std": step(global_std),
            "global_mean": global_mean.tolist(),
            "global_std": global_std.tolist(),
        }

    return {
        "state": {"default": one(state)},
        "action": {"default": one(action)},
        "num_episodes": len(states),
        "num_transition": int(sum(len(x) for x in states)),
    }


def write_video(path: Path, frames: np.ndarray, fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        path,
        fps=fps,
        codec="libx264",
        macro_block_size=1,
        ffmpeg_params=["-pix_fmt", "yuv420p", "-g", "1"],
    )
    try:
        for frame in frames:
            writer.append_data(frame)
    finally:
        writer.close()


def write_parquet(path: Path, state: np.ndarray, action: np.ndarray, ep: Episode, fps: int, global_offset: int) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    length = len(state)
    frame_index = np.arange(length, dtype=np.int64)
    table = pa.table(
        {
            "observation.state": pa.array(state.astype(np.float32).tolist(), type=pa.list_(pa.float32())),
            "action": pa.array(action.astype(np.float32).tolist(), type=pa.list_(pa.float32())),
            "timestamp": pa.array((frame_index / fps).astype(np.float32)),
            "frame_index": pa.array(frame_index),
            "episode_index": pa.array(np.full(length, ep.output_index, dtype=np.int64)),
            "index": pa.array(np.arange(global_offset, global_offset + length, dtype=np.int64)),
            "task_index": pa.array(np.full(length, ep.task_index, dtype=np.int64)),
        }
    )
    pq.write_table(table, path)
    return global_offset + length


def load_episode(input_root: Path, source_path: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    with h5py.File(source_path, "r") as h5:
        qpos = np.asarray(h5["joint_action/vector"], dtype=np.float32)
        images = {out_key: read_camera_frames(h5, in_key) for in_key, out_key in CAMERA_MAP.items()}

    min_len = min([len(qpos), *(len(v) for v in images.values())])
    qpos = qpos[:min_len]
    images = {key: value[:min_len] for key, value in images.items()}
    if min_len < 2:
        raise ValueError(f"{source_path} has fewer than 2 frames after alignment")
    if qpos.shape[1] != 14:
        raise ValueError(f"{source_path} qpos dim should be 14, got {qpos.shape}")
    return qpos, images


def build_episodes(args: argparse.Namespace) -> list[Episode]:
    files = sorted((args.input_root / "data").glob("episode*.hdf5"), key=episode_number)
    if args.max_episodes is not None:
        files = files[: args.max_episodes]
    if not files:
        raise FileNotFoundError(f"No episode*.hdf5 found under {args.input_root / 'data'}")

    fallback = args.default_task
    if fallback is None:
        fallback = args.input_root.parent.name.replace("_", " ")

    episodes = []
    task_to_index: dict[str, int] = {}
    for output_idx, path in enumerate(files):
        source_idx = episode_number(path)
        task = read_instruction(
            input_root=args.input_root,
            ep_idx=source_idx,
            split=args.instruction_split,
            instruction_index=args.instruction_index,
            fallback=fallback,
        )
        if task not in task_to_index:
            task_to_index[task] = len(task_to_index)
        episodes.append(
            Episode(
                source_path=path,
                source_index=source_idx,
                output_index=output_idx,
                task=task,
                task_index=task_to_index[task],
            )
        )
    return episodes


def main() -> None:
    args = parse_args()
    if args.output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"{args.output_root} already exists; pass --overwrite to replace it")
        shutil.rmtree(args.output_root)

    episodes = build_episodes(args)
    args.output_root.mkdir(parents=True, exist_ok=True)

    tasks = sorted({ep.task_index: ep.task for ep in episodes}.items())
    task_rows = [{"task_index": idx, "task": task} for idx, task in tasks]
    episode_rows: list[dict[str, Any]] = []
    episode_stats_rows: list[dict[str, Any]] = []
    all_states: list[np.ndarray] = []
    all_actions: list[np.ndarray] = []
    global_offset = 0
    first_shape: tuple[int, int, int] | None = None

    for ep in episodes:
        qpos, images = load_episode(args.input_root, ep.source_path)
        state = qpos
        action = np.concatenate([qpos[1:], qpos[-1:]], axis=0).astype(np.float32)
        chunk = ep.output_index // args.chunks_size

        parquet_path = args.output_root / f"data/chunk-{chunk:03d}/episode_{ep.output_index:06d}.parquet"
        global_offset = write_parquet(parquet_path, state, action, ep, args.fps, global_offset)

        for video_key, frames in images.items():
            if first_shape is None:
                first_shape = tuple(frames.shape[1:])
            video_path = (
                args.output_root
                / f"videos/chunk-{chunk:03d}/observation.images.{video_key}/episode_{ep.output_index:06d}.mp4"
            )
            write_video(video_path, frames, args.fps)

        episode_rows.append(
            {
                "episode_index": ep.output_index,
                "tasks": [ep.task],
                "length": int(len(state)),
                "source_episode_index": ep.source_index,
                "source_path": str(ep.source_path),
            }
        )
        columns = {
            "observation.state": state,
            "action": action,
            "timestamp": (np.arange(len(state), dtype=np.float32) / args.fps),
            "frame_index": np.arange(len(state), dtype=np.int64),
            "episode_index": np.full(len(state), ep.output_index, dtype=np.int64),
            "index": np.arange(global_offset - len(state), global_offset, dtype=np.int64),
            "task_index": np.full(len(state), ep.task_index, dtype=np.int64),
        }
        episode_stats_rows.append({"episode_index": ep.output_index, "stats": episode_lerobot_stats(columns)})
        all_states.append(state)
        all_actions.append(action)
        print(f"[{ep.output_index + 1}/{len(episodes)}] converted {ep.source_path}")

    if first_shape is None:
        raise RuntimeError("No camera frames were written")
    height, width, channels = first_shape
    if channels != 3:
        raise ValueError(f"Expected RGB frames, got shape {first_shape}")

    features = {
        "observation.state": numeric_feature("float32", [14], [JOINT_NAMES]),
        "action": numeric_feature("float32", [14], [JOINT_NAMES]),
        "observation.images.cam_high": video_feature(height, width, args.fps),
        "observation.images.cam_left_wrist": video_feature(height, width, args.fps),
        "observation.images.cam_right_wrist": video_feature(height, width, args.fps),
        "timestamp": numeric_feature("float32", [1]),
        "frame_index": numeric_feature("int64", [1]),
        "episode_index": numeric_feature("int64", [1]),
        "index": numeric_feature("int64", [1]),
        "task_index": numeric_feature("int64", [1]),
    }
    info = {
        "codebase_version": "v2.1",
        "robot_type": "aloha-agilex",
        "total_episodes": len(episodes),
        "total_frames": int(sum(row["length"] for row in episode_rows)),
        "total_tasks": len(task_rows),
        "total_videos": len(episodes) * len(CAMERA_MAP),
        "total_chunks": (len(episodes) + args.chunks_size - 1) // args.chunks_size,
        "chunks_size": args.chunks_size,
        "fps": args.fps,
        "splits": {"train": f"0:{len(episodes)}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": features,
    }

    write_json(args.output_root / "meta/info.json", info)
    write_jsonl(args.output_root / "meta/tasks.jsonl", task_rows)
    write_jsonl(args.output_root / "meta/episodes.jsonl", episode_rows)
    write_jsonl(args.output_root / "meta/episodes_stats.jsonl", episode_stats_rows)
    write_json(args.output_root / "meta/modality.json", {})
    write_json(args.output_root / "meta/embodiment.json", {"robot_type": "aloha-agilex"})
    write_json(args.output_root / "dataset_stats.json", fastwam_dataset_stats(all_states, all_actions))

    print(f"Converted {len(episodes)} episodes to {args.output_root}")
    print(f"FastWAM stats written to {args.output_root / 'dataset_stats.json'}")


if __name__ == "__main__":
    main()
