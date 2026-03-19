from __future__ import annotations

from collections import deque
import os
import pathlib
import sys
from typing import Any, Dict, List

import numpy as np
import torch
import yaml


ROOT_DIR = str(pathlib.Path(__file__).resolve().parent.parent)
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)
OAT_ROOT = str(pathlib.Path(ROOT_DIR) / "oat")
if OAT_ROOT not in sys.path:
    sys.path.append(OAT_ROOT)
LIBERO_ROOT = str(pathlib.Path(ROOT_DIR) / "oat" / "third_party" / "LIBERO")
if LIBERO_ROOT not in sys.path:
    sys.path.append(LIBERO_ROOT)


def _ensure_libero_config() -> None:
    """确保 LIBERO 运行配置存在并与当前仓库路径对齐。

    接口对应:
    - 输入接口: 读取模块级 `ROOT_DIR/LIBERO_ROOT`。
    - 输出接口: 设置 `LIBERO_CONFIG_PATH` 环境变量，并在缺失时生成 config.yaml。
    """
    libero_config_root = pathlib.Path(ROOT_DIR) / ".libero"
    libero_config_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("LIBERO_CONFIG_PATH", str(libero_config_root))

    config_path = libero_config_root / "config.yaml"
    if config_path.exists():
        return

    benchmark_root = pathlib.Path(LIBERO_ROOT) / "libero" / "libero"
    config = {
        "benchmark_root": str(benchmark_root),
        "bddl_files": str(benchmark_root / "bddl_files"),
        "init_states": str(benchmark_root / "init_files"),
        "datasets": str(pathlib.Path(LIBERO_ROOT) / "libero" / "datasets"),
        "assets": str(benchmark_root / "assets"),
    }
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f)


_ensure_libero_config()

from libero.libero import benchmark
from libero.libero import get_libero_path
from oat.oat.env.libero.env import LiberoEnv, task_name_to_suite_and_ids


def _build_state_tensor(obs: Dict[str, Any], state_keys: List[str], device: str) -> torch.Tensor:
    """将环境观测中的多路状态字段拼接成模型输入状态张量。

    参数:
        obs: 环境观测字典。
        state_keys: 需要提取并拼接的状态键顺序。
        device: 目标设备字符串。

    返回:
        形状为 `[1, state_dim]` 的 float32 状态张量。
    """
    pieces = []
    for key in state_keys:
        arr = np.asarray(obs[key], dtype=np.float32).reshape(-1)
        pieces.append(arr)
    state = np.concatenate(pieces, axis=0)
    return torch.from_numpy(state).to(device=device).unsqueeze(0)


def _initial_frame_window(frame: np.ndarray, num_frames: int) -> deque[np.ndarray]:
    """用当前帧初始化固定长度窗口，匹配流式视觉输入接口。

    参数:
        frame: 当前 RGB 帧。
        num_frames: 窗口长度。

    返回:
        长度为 `num_frames` 的双端队列，初值由同一帧重复填充。
    """
    q: deque[np.ndarray] = deque(maxlen=num_frames)
    for _ in range(num_frames):
        q.append(np.asarray(frame, dtype=np.uint8))
    return q


def _window_array(frame_window: deque[np.ndarray]) -> np.ndarray:
    """将帧窗口转换为连续数组，供 pipeline `push_observation` 调用。"""
    return np.stack(list(frame_window), axis=0)


def _set_init_state(env: LiberoEnv, init_state: np.ndarray) -> Dict[str, Any]:
    """设置环境初始状态并返回标准化观测。

    参数:
        env: LIBERO 环境实例。
        init_state: 要写入的初始状态向量。

    返回:
        经 `env._extract_obs` 规整后的观测字典。
    """
    raw_obs = env.env.set_init_state(init_state)
    env.done = False
    env.cur_step = 0
    return env._extract_obs(raw_obs)


def _load_task_init_states(task: Any) -> torch.Tensor:
    """加载任务对应的离线初始状态集合。

    参数:
        task: benchmark 任务对象，需包含 `problem_folder/init_states_file`。

    返回:
        任务初始化状态张量，供 rollout 选择起始场景。
    """
    init_states_path = pathlib.Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file
    return torch.load(init_states_path, weights_only=False)
