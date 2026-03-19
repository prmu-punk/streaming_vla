import json
import math
import os
import pathlib
import random
import sys
import os
# 添加项目根目录到 sys.path 以解决找不到 model.qwen3_vl 的问题
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from dataclasses import dataclass
from typing import Any, Dict, List

import hydra
import numpy as np
import torch
import yaml
from multiprocessing.reduction import ForkingPickler
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, dataloader
from tqdm.auto import tqdm
import wandb


default_collate_func = dataloader.default_collate


def default_collate_override(batch) -> Any:
    """覆盖默认 collate 以关闭 shared memory，避免多进程存储句柄问题。

    参数:
        batch: DataLoader 提供的原始样本批。

    返回:
        使用原始默认 collate 聚合后的批对象。
    """
    dataloader._use_shared_memory = False
    return default_collate_func(batch)


setattr(dataloader, "default_collate", default_collate_override)

for t in torch._storage_classes:
    if sys.version_info[0] == 2:
        if t in ForkingPickler.dispatch:
            del ForkingPickler.dispatch[t]
    else:
        if t in ForkingPickler._extra_reducers:
            del ForkingPickler._extra_reducers[t]

ROOT_DIR = str(pathlib.Path(__file__).parent.parent)
if ROOT_DIR in sys.path:
    sys.path.remove(ROOT_DIR)
sys.path.insert(0, ROOT_DIR)


def _normalize_hydra_cli_config_path(argv: List[str]) -> List[str]:
    normalized = list(argv)
    script_dir = pathlib.Path(__file__).resolve().parent
    root_dir = pathlib.Path(ROOT_DIR).resolve()
    root_parent = root_dir.parent

    def _resolve_existing_dir(path_text: str) -> str:
        p = pathlib.Path(path_text)
        if p.is_absolute() and p.is_dir():
            return str(p)
        candidates = [
            pathlib.Path.cwd() / p,
            root_dir / p,
            root_parent / p,
            script_dir / p,
        ]
        for candidate in candidates:
            if candidate.is_dir():
                return str(candidate.resolve())
        return path_text

    for i, arg in enumerate(normalized):
        if arg.startswith("--config-path="):
            value = arg.split("=", 1)[1]
            normalized[i] = f"--config-path={_resolve_existing_dir(value)}"
            return normalized
        if arg == "--config-path" and i + 1 < len(normalized):
            normalized[i + 1] = _resolve_existing_dir(normalized[i + 1])
            return normalized
    return normalized

from dataset.libero90_async_offline_context_dataset import LiberoOfflineContextDataset, offline_context_collate
from model.rtc_async import ActionExpertRunner, ActionExpertRunnerConfig, RTCPipelineState, RTCVLAEntry, rtc_velocity_loss, schedule_rtc_chunk
from model.rtc_async.adapters import ActionShapeSpec, KVShapeSpec, StateConditionAdapter, StateShapeSpec
from model.rtc_async.qwen3_stream import export_selected_kv_cache
from model.vla_qwen3_rtc import Qwen3RTCVLAEncoder


@dataclass
class RTCAsyncResolvedConfig:
    selected_layers: list[int]
    inference_delay: int
    execute_horizon: int
    simulated_delay: int
    action_expert: Dict[str, Any]


def set_seed(seed: int) -> None:
    """设置 Python/Numpy/PyTorch 随机种子，保证训练可复现。

    参数:
        seed: 统一随机种子。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str) -> None:
    """确保目录存在，不存在则递归创建。

    参数:
        path: 目标目录路径。
    """
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)


def resolve_workspace_path(path_str: str) -> str:
    """将配置中的相对路径解析为仓库工作区绝对路径。

    参数:
        path_str: 绝对或相对路径字符串。

    返回:
        解析后的绝对路径。
    """
    p = pathlib.Path(path_str)
    if p.is_absolute():
        return str(p)
    return str(pathlib.Path(ROOT_DIR) / p)


def load_rtc_async_config(config_path: str) -> RTCAsyncResolvedConfig:
    """加载 rtc_async 配置并映射到训练期结构化配置对象。

    参数:
        config_path: rtc_async 配置文件路径。

    返回:
        `RTCAsyncResolvedConfig`，供调度器与动作专家初始化使用。
    """
    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    stream = raw.get("stream", {}) or {}
    rtc = raw.get("rtc", {}) or {}
    action_expert = raw.get("action_expert", {}) or {}

    selected_layers = [int(x) for x in stream.get("selected_layers", [])]
    if not selected_layers:
        raise ValueError("rtc_async.stream.selected_layers must be non-empty.")

    return RTCAsyncResolvedConfig(
        selected_layers=selected_layers,
        inference_delay=int(rtc.get("inference_delay", 0)),
        execute_horizon=int(rtc.get("execute_horizon", 1)),
        simulated_delay=int(rtc.get("simulated_delay", 0)),
        action_expert=action_expert,
    )


def build_action_expert_runner(
    *,
    rtc_cfg: RTCAsyncResolvedConfig,
    state_dim: int,
    action_dim: int,
    horizon: int,
) -> ActionExpertRunner:
    """按数据集维度约束构建动作专家运行器。

    参数:
        rtc_cfg: 解析后的 rtc_async 配置。
        state_dim: 数据集状态维度。
        action_dim: 数据集动作维度。
        horizon: 数据集 chunk horizon。

    返回:
        维度对齐并通过校验的 `ActionExpertRunner` 实例。
    """
    action_cfg = dict(rtc_cfg.action_expert)
    action_cfg["state_dim"] = int(action_cfg.get("state_dim", state_dim))
    action_cfg["action_dim"] = int(action_cfg.get("action_dim", action_dim))
    action_cfg["horizon"] = int(action_cfg.get("horizon", horizon))

    if action_cfg["state_dim"] != state_dim:
        raise ValueError(f"rtc_async.action_expert.state_dim={action_cfg['state_dim']} but dataset state_dim={state_dim}")
    if action_cfg["action_dim"] != action_dim:
        raise ValueError(f"rtc_async.action_expert.action_dim={action_cfg['action_dim']} but dataset action_dim={action_dim}")
    if action_cfg["horizon"] != horizon:
        raise ValueError(f"rtc_async.action_expert.horizon={action_cfg['horizon']} but dataset horizon={horizon}")

    runner_cfg = ActionExpertRunnerConfig(**action_cfg)
    return ActionExpertRunner(runner_cfg)


def _stack_anchor_state(sample_list: List[Dict[str, Any]], device: str) -> torch.Tensor:
    """提取并堆叠批次 anchor_state，输出 `[B, Ds]` 状态张量。

    参数:
        sample_list: 离线样本字典列表。
        device: 目标设备。

    返回:
        堆叠后的 anchor 状态张量。
    """
    state = torch.stack([sample["anchor_state"] for sample in sample_list], dim=0)
    return state.to(device)


def _validate_protocol(
    *,
    target_chunk: torch.Tensor,
    state: torch.Tensor,
    kv_cache: list[tuple[torch.Tensor, torch.Tensor]],
    selected_layers: list[int],
) -> None:
    """校验动作/state/KV 条件是否满足 rtc_async 接口协议。

    接口对应:
    - `target_chunk` 对应动作监督输入 `[B,H,D]`。
    - `state` 对应动作专家条件输入 `[B,Ds]`。
    - `kv_cache` 层数需与 `selected_layers` 一致。
    """
    action_spec = ActionShapeSpec(horizon=int(target_chunk.shape[1]), action_dim=int(target_chunk.shape[2]))
    state_spec = StateShapeSpec(state_dim=int(state.shape[-1]))
    kv_spec = KVShapeSpec(n_layers=len(selected_layers))

    if target_chunk.dim() != 3:
        raise ValueError(f"Action chunk must be [B,H,D], got {tuple(target_chunk.shape)}")
    if state.dim() != 2:
        raise ValueError(f"State must be [B,Ds], got {tuple(state.shape)}")
    if state_spec.state_dim <= 0 or action_spec.action_dim <= 0 or action_spec.horizon <= 0:
        raise ValueError("Invalid shape spec values in rtc_async adapters.")
    if kv_cache and len(kv_cache) != kv_spec.n_layers:
        raise ValueError(f"Exported KV layers={len(kv_cache)} != selected_layers={kv_spec.n_layers}")


def run_epoch(
    *,
    vla: Qwen3RTCVLAEncoder,
    action_expert: ActionExpertRunner,
    state_adapter: StateConditionAdapter | None,
    rtc_entry: RTCVLAEntry,
    rtc_cfg: RTCAsyncResolvedConfig,
    scheduler: Any,
    pipeline_state: RTCPipelineState,
    dataloader: DataLoader,
    cfg: DictConfig,
    train: bool,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
) -> Dict[str, float]:
    """执行单个训练或验证 epoch，并返回聚合指标。

    接口对应:
    - 输入接口: `dataloader -> vla.forward_offline_context_batch -> action_expert`。
    - 输出接口: 返回 `rtc_loss/execute_mse/avg_delay` 等监控指标字典。

    参数:
        vla: VLM 条件编码器。
        action_expert: 动作专家网络。
        state_adapter: 可选状态适配器。
        rtc_entry: RTC 兼容入口封装。
        rtc_cfg: rtc_async 超参数。
        scheduler: RTC chunk 调度器。
        pipeline_state: 调度流水线状态容器。
        dataloader: 数据加载器。
        cfg: 训练总配置。
        train: True 为训练模式，False 为验证模式。
        optimizer: 训练时优化器。
        epoch: 当前 epoch 索引。

    返回:
        当前阶段（train/val）指标字典。
    """
    if train:
        action_expert.train()
        if state_adapter is not None:
            state_adapter.train()
        vla.train(bool(cfg.rtc_async.train_vla))
    else:
        action_expert.eval()
        if state_adapter is not None:
            state_adapter.eval()
        vla.eval()

    total_loss = 0.0
    total_exec_mse = 0.0
    total_delay = 0.0
    n_samples = 0
    n_updates = 0

    max_batches_key = "max_train_batches" if train else "max_val_batches"
    max_batches = cfg.training.get(max_batches_key, None) if train else cfg.eval.get(max_batches_key, None)

    progress = tqdm(
        dataloader,
        desc=f"{'train' if train else 'val'} epoch {epoch}",
        leave=False,
        dynamic_ncols=True,
    )

    for batch_idx, sample_list in enumerate(progress):
        if train:
            assert optimizer is not None
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train and bool(cfg.rtc_async.train_vla)):
            out = vla.forward_offline_context_batch(
                samples=sample_list,
                num_frames=int(cfg.model.num_frames),
                source_dt_ms=int(cfg.training.source_dt_ms),
                return_condition_cache=True,
            )

        target_chunk = out["target_chunk"]
        past_key_values = out.get("past_key_values", None)
        attention_mask = out.get("attention_mask", None)
        if target_chunk is None or past_key_values is None:
            continue

        all_kvs = past_key_values

        # 处理不同的 KV Cache 格式
        if isinstance(all_kvs, list) and len(all_kvs) > 0 and hasattr(all_kvs[0], "keys") and hasattr(all_kvs[0], "values"):
            all_kvs_list = [(c.keys, c.values) for c in all_kvs]
        elif hasattr(all_kvs, "to_legacy_cache"):
            all_kvs_list = all_kvs.to_legacy_cache()
        elif type(all_kvs).__name__ == "DynamicCache":
            if hasattr(all_kvs, "key_cache"):
                all_kvs_list = list(zip(all_kvs.key_cache, all_kvs.value_cache))
            else:
                all_kvs_list = list(all_kvs)
        elif isinstance(all_kvs, tuple) and isinstance(all_kvs[0], tuple):
            all_kvs_list = list(all_kvs)
        else:
            raise ValueError(f"未知的 KV Cache 格式: {type(all_kvs)}")

        # 导出指定层的 KV
        # 如果配置文件没有指定，默认取最后3层
        if hasattr(rtc_cfg, "selected_layers") and rtc_cfg.selected_layers:
            selected_kvs = [all_kvs_list[i] for i in rtc_cfg.selected_layers]
        else:
            selected_kvs = all_kvs_list[-3:] if len(all_kvs_list) >= 3 else all_kvs_list
        
        # 格式化给 Action Expert
        kv_cache = []
        for layer_kv in selected_kvs:
            if hasattr(layer_kv, "keys") and hasattr(layer_kv, "values"):
                kv_cache.append((layer_kv.keys, layer_kv.values))
            elif isinstance(layer_kv, tuple) and len(layer_kv) == 2:
                kv_cache.append(layer_kv)
            elif isinstance(layer_kv, tuple) and len(layer_kv) == 3 and layer_kv[2] is None:
                kv_cache.append((layer_kv[0], layer_kv[1]))
            elif isinstance(layer_kv, tuple) and hasattr(layer_kv[0], "keys"):
                kv_cache.append((layer_kv[0].keys, layer_kv[0].values))
            else:
                raise ValueError(f"层 KV 格式不支持拆包: {type(layer_kv)}")

        state = _stack_anchor_state(sample_list, vla.device)
        if state_adapter is not None:
            state = state_adapter(state)

        _validate_protocol(
            target_chunk=target_chunk,
            state=state,
            kv_cache=kv_cache,
            selected_layers=rtc_cfg.selected_layers,
        )

        rtc_batch = rtc_entry.build_training_rtc_batch(
            action_chunk=target_chunk,
            simulated_delay=rtc_cfg.simulated_delay,
        )

        if not bool(cfg.rtc_async.use_attention_mask):
            attention_mask = None

        pred_u_t = action_expert(
            noisy_action=rtc_batch.x_t,
            state=state,
            time=rtc_batch.time,
            kv_cache=kv_cache,
            attention_mask=attention_mask,
        )
        loss = rtc_velocity_loss(pred_u_t=pred_u_t, batch=rtc_batch)

        if train:
            loss.backward()
            if cfg.training.max_grad_norm is not None:
                params = list(action_expert.parameters())
                if state_adapter is not None:
                    params += list(state_adapter.parameters())
                if bool(cfg.rtc_async.train_vla):
                    params += list(vla.parameters())
                torch.nn.utils.clip_grad_norm_(params, float(cfg.training.max_grad_norm))
            assert optimizer is not None
            optimizer.step()
            n_updates += 1

        with torch.no_grad():
            sampled_chunk = None
            if int(cfg.rtc_async.sample_eval_every) > 0 and (batch_idx % int(cfg.rtc_async.sample_eval_every) == 0):
                sampled_chunk = action_expert.sample(
                    state=state,
                    kv_cache=kv_cache,
                    attention_mask=attention_mask,
                    kv_cache_key=("train" if train else "val", epoch, batch_idx),
                )

            execute_mse = 0.0
            if sampled_chunk is not None:
                step_id = pipeline_state.next_id()
                chunk_packet = schedule_rtc_chunk(
                    scheduler=scheduler,
                    step_id=step_id,
                    next_chunk=sampled_chunk,
                    inference_delay=rtc_cfg.inference_delay,
                    execute_horizon=rtc_cfg.execute_horizon,
                )
                pipeline_state.chunk_queue.append(chunk_packet)
                popped = pipeline_state.pop_action_chunk()
                if popped is not None:
                    target_exec = target_chunk[:, : popped.execute_chunk.shape[1]]
                    execute_mse = float(torch.nn.functional.mse_loss(popped.execute_chunk, target_exec).item())

            bs = int(target_chunk.shape[0])
            total_loss += float(loss.detach().item()) * bs
            total_exec_mse += float(execute_mse) * bs
            total_delay += float(rtc_batch.delay.float().mean().item()) * bs
            n_samples += bs

            progress.set_postfix(
                rtc_loss=f"{(total_loss / max(n_samples, 1)):.4f}",
                exec_mse=f"{(total_exec_mse / max(n_samples, 1)):.4f}",
                samples=n_samples,
            )

        if max_batches is not None and (batch_idx + 1) >= int(max_batches):
            break

    denom = max(n_samples, 1)
    prefix = "train" if train else "val"
    return {
        f"{prefix}/rtc_loss": total_loss / denom,
        f"{prefix}/execute_mse": total_exec_mse / denom,
        f"{prefix}/avg_delay": total_delay / denom,
        f"{prefix}/samples": float(n_samples),
        f"{prefix}/updates": float(n_updates),
    }


def save_checkpoint(
    path: str,
    *,
    action_expert: ActionExpertRunner,
    state_adapter: StateConditionAdapter | None,
    vla: Qwen3RTCVLAEncoder,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    cfg: DictConfig,
) -> None:
    """保存训练检查点，包含模型权重、优化器和配置快照。

    参数:
        path: 输出 checkpoint 路径。
        action_expert: 动作专家模型。
        state_adapter: 可选状态适配器。
        vla: VLA 编码器。
        optimizer: 优化器实例。
        epoch: 当前 epoch。
        global_step: 全局步数。
        cfg: 训练配置对象。
    """
    payload = {
        "epoch": int(epoch),
        "global_step": int(global_step),
        "action_expert": action_expert.state_dict(),
        "optimizer": optimizer.state_dict(),
        "cfg": OmegaConf.to_container(cfg, resolve=True),
    }
    if state_adapter is not None:
        payload["state_adapter"] = state_adapter.state_dict()
    if bool(cfg.rtc_async.train_vla):
        payload["vla"] = vla.state_dict()
    torch.save(payload, path)


def load_checkpoint(
    path: str,
    *,
    action_expert: ActionExpertRunner,
    state_adapter: StateConditionAdapter | None,
    vla: Qwen3RTCVLAEncoder,
    optimizer: torch.optim.Optimizer | None = None,
    train_vla: bool,
) -> tuple[int, int]:
    """加载检查点并恢复训练状态。

    返回:
        `(epoch, global_step)`，用于恢复训练游标。
    """
    payload = torch.load(path, map_location=vla.device)
    action_expert.load_state_dict(payload["action_expert"], strict=True)
    if state_adapter is not None and "state_adapter" in payload:
        state_adapter.load_state_dict(payload["state_adapter"], strict=True)
    if train_vla and "vla" in payload:
        vla.load_state_dict(payload["vla"], strict=True)
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    epoch = int(payload.get("epoch", -1))
    global_step = int(payload.get("global_step", 0))
    return epoch, global_step


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent / "configs"),
    config_name="train_libero90_async",
)
def main(cfg: DictConfig) -> None:
    """离线训练入口：构建数据、模型、优化器并驱动 epoch 循环。

    参数:
        cfg: Hydra 注入的训练配置。
    """
    OmegaConf.resolve(cfg)
    set_seed(int(cfg.training.seed))

    out_dir = os.getcwd()
    ensure_dir(out_dir)
    ensure_dir(os.path.join(out_dir, "checkpoints"))

    wandb.init(
        project="streaming_vla",
        name=pathlib.Path(out_dir).name,
        dir=out_dir,
        mode="offline",
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    rtc_config_path = resolve_workspace_path(str(cfg.rtc_async.config_path))
    rtc_cfg = load_rtc_async_config(rtc_config_path)
    rtc_entry = RTCVLAEntry(config_path=rtc_config_path)

    vla = Qwen3RTCVLAEncoder(config_path=resolve_workspace_path(str(cfg.model.vla_config_path)))

    if cfg.dataset.get("chunk_horizon", None) is not None:
        chunk_horizon = int(cfg.dataset.chunk_horizon)
    else:
        chunk_horizon = int(rtc_cfg.action_expert.get("horizon", 0))
    if chunk_horizon <= 0:
        raise ValueError(
            "chunk_horizon is unresolved. Set dataset.chunk_horizon or rtc_async.action_expert.horizon > 0."
        )

    anchor_stride = cfg.dataset.get("anchor_stride_steps", None)
    if anchor_stride is None:
        anchor_stride = 1
    anchor_stride = int(anchor_stride)

    base_ep = LiberoOfflineContextDataset(
        zarr_path=resolve_workspace_path(str(cfg.dataset.zarr_path)),
        image_key=str(cfg.dataset.image_key),
        action_key=str(cfg.dataset.action_key),
        state_keys=[str(k) for k in cfg.dataset.state_keys],
        prompt_key=str(cfg.dataset.prompt_key),
        source_dt_ms=int(cfg.training.source_dt_ms),
        step_dt_min_ms=int(cfg.training.step_dt_min_ms),
        step_dt_max_ms=int(cfg.training.step_dt_max_ms),
        num_frames=int(cfg.model.num_frames),
        chunk_horizon=int(chunk_horizon),
        anchor_stride_steps=anchor_stride,
        max_context_len=int(float(cfg.model.max_context_len)),
        context_budget_tokens=int(cfg.model.context_budget_tokens),
        max_episodes=cfg.dataset.max_episodes,
    )

    n_episodes = len(base_ep.base)
    val_ratio = float(cfg.dataset.val_ratio)
    if not (0.0 < val_ratio < 1.0):
        raise ValueError(f"val_ratio must be in (0,1), got {val_ratio}")
    n_val_eps = max(1, int(round(n_episodes * val_ratio)))
    n_train_eps = max(1, n_episodes - n_val_eps)
    if n_train_eps + n_val_eps != n_episodes:
        n_val_eps = n_episodes - n_train_eps

    perm = torch.randperm(n_episodes, generator=torch.Generator().manual_seed(int(cfg.training.seed))).tolist()
    train_eps = perm[:n_train_eps]
    val_eps = perm[n_train_eps:]

    train_set = LiberoOfflineContextDataset(
        zarr_path=resolve_workspace_path(str(cfg.dataset.zarr_path)),
        image_key=str(cfg.dataset.image_key),
        action_key=str(cfg.dataset.action_key),
        state_keys=[str(k) for k in cfg.dataset.state_keys],
        prompt_key=str(cfg.dataset.prompt_key),
        source_dt_ms=int(cfg.training.source_dt_ms),
        step_dt_min_ms=int(cfg.training.step_dt_min_ms),
        step_dt_max_ms=int(cfg.training.step_dt_max_ms),
        num_frames=int(cfg.model.num_frames),
        chunk_horizon=int(chunk_horizon),
        anchor_stride_steps=anchor_stride,
        max_context_len=int(float(cfg.model.max_context_len)),
        context_budget_tokens=int(cfg.model.context_budget_tokens),
        max_episodes=cfg.dataset.max_episodes,
        episode_indices=train_eps,
    )
    val_set = LiberoOfflineContextDataset(
        zarr_path=resolve_workspace_path(str(cfg.dataset.zarr_path)),
        image_key=str(cfg.dataset.image_key),
        action_key=str(cfg.dataset.action_key),
        state_keys=[str(k) for k in cfg.dataset.state_keys],
        prompt_key=str(cfg.dataset.prompt_key),
        source_dt_ms=int(cfg.training.source_dt_ms),
        step_dt_min_ms=int(cfg.training.step_dt_min_ms),
        step_dt_max_ms=int(cfg.training.step_dt_max_ms),
        num_frames=int(cfg.model.num_frames),
        chunk_horizon=int(chunk_horizon),
        anchor_stride_steps=anchor_stride,
        max_context_len=int(float(cfg.model.max_context_len)),
        context_budget_tokens=int(cfg.model.context_budget_tokens),
        max_episodes=cfg.dataset.max_episodes,
        episode_indices=val_eps,
    )

    if int(cfg.model.state_dim) != train_set.state_dim:
        raise ValueError(
            f"state_dim mismatch: model expects {cfg.model.state_dim}, dataset provides {train_set.state_dim}."
        )

    if int(chunk_horizon) != train_set.chunk_horizon:
        raise ValueError(
            f"chunk_horizon mismatch: inferred={chunk_horizon}, dataset={train_set.chunk_horizon}."
        )

    action_expert = build_action_expert_runner(
        rtc_cfg=rtc_cfg,
        state_dim=train_set.state_dim,
        action_dim=train_set.action_dim,
        horizon=train_set.chunk_horizon,
    ).to(vla.device)

    state_adapter: StateConditionAdapter | None = None
    if bool(cfg.rtc_async.use_state_adapter):
        state_adapter = StateConditionAdapter(train_set.state_dim, train_set.state_dim).to(vla.device)

    scheduler = rtc_entry.build_scheduler(
        horizon=train_set.chunk_horizon,
        action_dim=train_set.action_dim,
        device=torch.device(vla.device),
    )
    train_pipeline = RTCPipelineState()
    val_pipeline = RTCPipelineState()

    train_loader = DataLoader(train_set, collate_fn=offline_context_collate, **cfg.dataloader)
    val_loader = DataLoader(val_set, collate_fn=offline_context_collate, **cfg.val_dataloader)

    params = list(action_expert.parameters())
    if state_adapter is not None:
        params += list(state_adapter.parameters())
    if bool(cfg.rtc_async.train_vla):
        params += list(vla.parameters())

    optimizer = torch.optim.AdamW(
        params,
        lr=float(cfg.optimizer.lr),
        betas=(float(cfg.optimizer.beta1), float(cfg.optimizer.beta2)),
        weight_decay=float(cfg.optimizer.weight_decay),
    )

    log_path = os.path.join(out_dir, "train_log.jsonl")
    best_key = str(cfg.checkpoint.best_monitor)
    best_val = math.inf if str(cfg.checkpoint.best_mode) == "min" else -math.inf

    global_step = 0
    start_epoch = 0
    resume_from = cfg.checkpoint.get("resume_from", None)
    if resume_from:
        resumed_epoch, global_step = load_checkpoint(
            str(resume_from),
            action_expert=action_expert,
            state_adapter=state_adapter,
            vla=vla,
            optimizer=optimizer,
            train_vla=bool(cfg.rtc_async.train_vla),
        )
        start_epoch = resumed_epoch + 1

    for epoch in range(start_epoch, int(cfg.training.num_epochs)):
        train_log = run_epoch(
            vla=vla,
            action_expert=action_expert,
            state_adapter=state_adapter,
            rtc_entry=rtc_entry,
            rtc_cfg=rtc_cfg,
            scheduler=scheduler,
            pipeline_state=train_pipeline,
            dataloader=train_loader,
            cfg=cfg,
            train=True,
            optimizer=optimizer,
            epoch=epoch,
        )
        val_log = run_epoch(
            vla=vla,
            action_expert=action_expert,
            state_adapter=state_adapter,
            rtc_entry=rtc_entry,
            rtc_cfg=rtc_cfg,
            scheduler=scheduler,
            pipeline_state=val_pipeline,
            dataloader=val_loader,
            cfg=cfg,
            train=False,
            optimizer=None,
            epoch=epoch,
        )

        metrics: Dict[str, float] = {**train_log, **val_log}
        metrics["epoch"] = float(epoch)
        metrics["global_step"] = float(global_step)

        print(json.dumps(metrics, ensure_ascii=False))
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(metrics, ensure_ascii=False) + "\n")
        wandb.log(metrics, step=global_step)

        if epoch % int(cfg.checkpoint.save_every) == 0:
            ep_path = os.path.join(out_dir, "checkpoints", f"epoch_{epoch:04d}.pt")
            save_checkpoint(
                ep_path,
                action_expert=action_expert,
                state_adapter=state_adapter,
                vla=vla,
                optimizer=optimizer,
                epoch=epoch,
                global_step=global_step,
                cfg=cfg,
            )
            save_checkpoint(
                os.path.join(out_dir, "checkpoints", "latest.pt"),
                action_expert=action_expert,
                state_adapter=state_adapter,
                vla=vla,
                optimizer=optimizer,
                epoch=epoch,
                global_step=global_step,
                cfg=cfg,
            )

        if best_key in metrics:
            cur = float(metrics[best_key])
            better = cur < best_val if str(cfg.checkpoint.best_mode) == "min" else cur > best_val
            if better:
                best_val = cur
                save_checkpoint(
                    os.path.join(out_dir, "checkpoints", "best.pt"),
                    action_expert=action_expert,
                    state_adapter=state_adapter,
                    vla=vla,
                    optimizer=optimizer,
                    epoch=epoch,
                    global_step=global_step,
                    cfg=cfg,
                )

        global_step += int(train_log["train/updates"])

    wandb.finish()


if __name__ == "__main__":
    sys.argv = _normalize_hydra_cli_config_path(sys.argv)
    os.chdir(ROOT_DIR)
    main()
