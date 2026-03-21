import json
import math
import os
import pathlib
import random
import sys
from dataclasses import dataclass
from typing import Any, Dict, List

import hydra
import numpy as np
import torch
import yaml
from accelerate import Accelerator
from multiprocessing.reduction import ForkingPickler
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, dataloader
from tqdm.auto import tqdm
import wandb
default_collate_func = dataloader.default_collate
def default_collate_override(batch) -> Any:
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
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from dataset.bucket_sampler import BucketBatchSampler
from dataset.libero90_async_offline_context_dataset import LiberoOfflineContextDataset, offline_context_collate
from model.rtc_async.action_expert.runner import ActionExpertRunner, ActionExpertRunnerConfig
from model.rtc_async.pipeline.scheduler import RTCChunkScheduler
from model.rtc_async.qwen3_stream.kv_export import export_selected_kv_cache
from model.rtc_async.training.loss_rtc import build_rtc_inpainting_batch, rtc_velocity_loss
from model.vla_qwen3_rtc import Qwen3RTCVLAEncoder


@dataclass
class RTCAsyncResolvedConfig:
    selected_layers: list[int]
    inference_delay: int
    execute_horizon: int
    simulated_delay: int
    action_expert: Dict[str, Any]
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
def ensure_dir(path: str) -> None:
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)


def resolve_workspace_path(path_str: str) -> str:
    p = pathlib.Path(path_str)
    if p.is_absolute():
        return str(p)
    return str(pathlib.Path(ROOT_DIR) / p)
def load_rtc_async_config(config_path: str) -> RTCAsyncResolvedConfig:
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


def run_epoch(
    *,
    vla: Qwen3RTCVLAEncoder,
    action_expert: ActionExpertRunner,
    rtc_cfg: RTCAsyncResolvedConfig,
    scheduler: RTCChunkScheduler,
    dataloader: DataLoader,
    cfg: DictConfig,
    train: bool,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    accelerator: Accelerator,
) -> Dict[str, float]:
    """执行单个训练或验证 epoch，并返回聚合指标。

    接口对应:
    - 输入接口: `dataloader -> vla.forward_offline_context_batch -> action_expert`。
    - 输出接口: 返回 `rtc_loss/execute_mse/avg_delay` 等监控指标字典。
    """
    train_vla = bool(cfg.rtc_async.train_vla)
    if train:
        action_expert.train()
        vla.train(train_vla)
    else:
        action_expert.eval()
        vla.eval()

    total_loss = 0.0
    total_exec_mse = 0.0
    total_delay = 0.0
    n_samples = 0
    n_updates = 0

    max_batches_key = "max_train_batches" if train else "max_val_batches"
    max_batches = cfg.training.get(max_batches_key, None) if train else cfg.eval.get(max_batches_key, None)

    if accelerator.is_local_main_process:
        progress = tqdm(
            dataloader,
            desc=f"{'train' if train else 'val'} epoch {epoch}",
            leave=False,
            dynamic_ncols=True,
        )
    else:
        progress = dataloader

    for batch_idx, sample_list in enumerate(progress):
        if train:
            assert optimizer is not None
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train and train_vla):
            out = vla(
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

        kv_cache = export_selected_kv_cache(
            past_key_values=past_key_values,
            selected_layers=rtc_cfg.selected_layers,
            clone=False,
        )

        state = _stack_anchor_state(sample_list, accelerator.device)
        if target_chunk.dim() != 3:
            raise ValueError(f"Action chunk must be [B,H,D], got {tuple(target_chunk.shape)}")
        if state.dim() != 2:
            raise ValueError(f"State must be [B,Ds], got {tuple(state.shape)}")
        if kv_cache and len(kv_cache) != len(rtc_cfg.selected_layers):
            raise ValueError(
                f"Exported KV layers={len(kv_cache)} != selected_layers={len(rtc_cfg.selected_layers)}"
            )

        rtc_batch = build_rtc_inpainting_batch(
            action=target_chunk,
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
            accelerator.backward(loss)
            if cfg.training.max_grad_norm is not None and accelerator.sync_gradients:
                params = list(action_expert.parameters())
                if train_vla:
                    params += list(vla.parameters())
                accelerator.clip_grad_norm_(params, float(cfg.training.max_grad_norm))
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
                execute_chunk, _ = scheduler.schedule(
                    next_chunk=sampled_chunk,
                    inference_delay=rtc_cfg.inference_delay,
                    execute_horizon=rtc_cfg.execute_horizon,
                )
                target_exec = target_chunk[:, : execute_chunk.shape[1]]
                execute_mse = float(torch.nn.functional.mse_loss(execute_chunk, target_exec).item())

            bs = int(target_chunk.shape[0])
            total_loss += float(loss.detach().item()) * bs
            total_exec_mse += float(execute_mse) * bs
            total_delay += float(rtc_batch.delay.float().mean().item()) * bs
            n_samples += bs

            if accelerator.is_local_main_process and hasattr(progress, "set_postfix"):
                progress.set_postfix(
                    rtc_loss=f"{(total_loss / max(n_samples, 1)):.4f}",
                    exec_mse=f"{(total_exec_mse / max(n_samples, 1)):.4f}",
                    samples=n_samples,
                )

        if max_batches is not None and (batch_idx + 1) >= int(max_batches):
            break

    local_stats = torch.tensor(
        [[total_loss, total_exec_mse, total_delay, float(n_samples), float(n_updates)]],
        device=accelerator.device,
        dtype=torch.float64,
    )
    gathered_stats = accelerator.gather(local_stats)
    summed_stats = gathered_stats[:, :4].sum(dim=0)
    max_updates = gathered_stats[:, 4].max()

    prefix = "train" if train else "val"
    return {
        f"{prefix}/rtc_loss": float(summed_stats[0].item() / max(summed_stats[3].item(), 1.0)),
        f"{prefix}/execute_mse": float(summed_stats[1].item() / max(summed_stats[3].item(), 1.0)),
        f"{prefix}/avg_delay": float(summed_stats[2].item() / max(summed_stats[3].item(), 1.0)),
        f"{prefix}/samples": float(summed_stats[3].item()),
        f"{prefix}/updates": float(max_updates.item()),
    }


def save_checkpoint(
    path: str,
    *,
    action_expert: ActionExpertRunner,
    vla: Qwen3RTCVLAEncoder,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    cfg: DictConfig,
    accelerator: Accelerator,
) -> None:
    """保存训练检查点，包含模型权重、优化器和配置快照。

    参数:
        path: 输出 checkpoint 路径。
        action_expert: 动作专家模型。
        vla: VLA 编码器。
        optimizer: 优化器实例。
        epoch: 当前 epoch。
        global_step: 全局步数。
        cfg: 训练配置对象。
    """
    payload = {
        "epoch": int(epoch),
        "global_step": int(global_step),
        "action_expert": accelerator.get_state_dict(action_expert),
        "optimizer": optimizer.state_dict(),
        "cfg": OmegaConf.to_container(cfg, resolve=True),
    }
    if bool(cfg.rtc_async.train_vla):
        payload["vla"] = accelerator.get_state_dict(vla)
    accelerator.save(payload, path)


def load_checkpoint(
    path: str,
    *,
    action_expert: ActionExpertRunner,
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
    accelerator = Accelerator()
    set_seed(int(cfg.training.seed))

    out_dir = os.getcwd()
    if accelerator.is_main_process:
        ensure_dir(out_dir)
        ensure_dir(os.path.join(out_dir, "checkpoints"))
        wandb.init(
            project="streaming_vla",
            name=pathlib.Path(out_dir).name,
            dir=out_dir,
            mode="offline",
            config=OmegaConf.to_container(cfg, resolve=True),
        )
    accelerator.wait_for_everyone()

    rtc_config_path = resolve_workspace_path(str(cfg.rtc_async.config_path))
    rtc_cfg = load_rtc_async_config(rtc_config_path)

    vla = Qwen3RTCVLAEncoder(config_path=resolve_workspace_path(str(cfg.model.vla_config_path)))
    vla.device = str(accelerator.device)
    vla.model.to(accelerator.device)
    vla.state_encoder.to(accelerator.device)

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

    full_set = LiberoOfflineContextDataset(
        zarr_path=resolve_workspace_path(str(cfg.dataset.zarr_path)),
        image_key=str(cfg.dataset.image_key),
        aux_image_key=str(cfg.dataset.aux_image_key) if cfg.dataset.get("aux_image_key", None) else None,
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
        episode_cache_size=int(cfg.dataset.get("episode_cache_size", 8)),
        max_episodes=cfg.dataset.max_episodes,
        processor=vla.processor,
        state_placeholder_token=vla.state_placeholder_token,
    )

    n_episodes = len(full_set.base)
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

    train_indices = full_set.sample_indices_for_episodes(train_eps)
    val_indices = full_set.sample_indices_for_episodes(val_eps)

    if int(cfg.model.state_dim) != full_set.state_dim:
        raise ValueError(
            f"state_dim mismatch: model expects {cfg.model.state_dim}, dataset provides {full_set.state_dim}."
        )

    if int(chunk_horizon) != full_set.chunk_horizon:
        raise ValueError(
            f"chunk_horizon mismatch: inferred={chunk_horizon}, dataset={full_set.chunk_horizon}."
        )

    action_expert = build_action_expert_runner(
        rtc_cfg=rtc_cfg,
        state_dim=full_set.state_dim,
        action_dim=full_set.action_dim,
        horizon=full_set.chunk_horizon,
    ).to(accelerator.device)

    scheduler = RTCChunkScheduler(
        horizon=full_set.chunk_horizon,
        action_dim=full_set.action_dim,
        device=torch.device(accelerator.device),
    )

    train_bs = int(cfg.dataloader.batch_size)
    val_bs = int(cfg.val_dataloader.batch_size)
    all_lengths = [full_set.get_estimated_length(i) for i in range(len(full_set))]

    train_bucket_sampler = BucketBatchSampler(
        all_lengths,
        batch_size=train_bs,
        shuffle=bool(cfg.dataloader.shuffle),
        drop_last=bool(cfg.dataloader.drop_last),
        indices=train_indices,
        seed=int(cfg.training.seed),
    )
    val_bucket_sampler = BucketBatchSampler(
        all_lengths,
        batch_size=val_bs,
        shuffle=False,
        drop_last=bool(cfg.val_dataloader.drop_last),
        indices=val_indices,
        seed=int(cfg.training.seed),
    )
    train_loader = DataLoader(
        full_set,
        batch_sampler=train_bucket_sampler,
        collate_fn=offline_context_collate,
        num_workers=int(cfg.dataloader.num_workers),
        pin_memory=bool(cfg.dataloader.pin_memory),
        persistent_workers=bool(cfg.dataloader.get("persistent_workers", int(cfg.dataloader.num_workers) > 0)),
    )
    val_loader = DataLoader(
        full_set,
        batch_sampler=val_bucket_sampler,
        collate_fn=offline_context_collate,
        num_workers=int(cfg.val_dataloader.num_workers),
        pin_memory=bool(cfg.val_dataloader.pin_memory),
        persistent_workers=bool(cfg.val_dataloader.get("persistent_workers", int(cfg.val_dataloader.num_workers) > 0)),
    )

    params = list(action_expert.parameters())
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
            vla=vla,
            optimizer=optimizer,
            train_vla=bool(cfg.rtc_async.train_vla),
        )
        start_epoch = resumed_epoch + 1
        accelerator.print(
            json.dumps(
                {
                    "resume_from": str(resume_from),
                    "resumed_epoch": int(resumed_epoch),
                    "start_epoch": int(start_epoch),
                    "global_step": int(global_step),
                },
                ensure_ascii=False,
            )
        )

    if bool(cfg.rtc_async.train_vla):
        vla, action_expert, optimizer, train_loader, val_loader = accelerator.prepare(
            vla,
            action_expert,
            optimizer,
            train_loader,
            val_loader,
        )
    else:
        action_expert, optimizer, train_loader, val_loader = accelerator.prepare(
            action_expert,
            optimizer,
            train_loader,
            val_loader,
        )

    for epoch in range(start_epoch, int(cfg.training.num_epochs)):
        train_bucket_sampler.set_epoch(epoch)
        train_log = run_epoch(
            vla=vla,
            action_expert=action_expert,
            rtc_cfg=rtc_cfg,
            scheduler=scheduler,
            dataloader=train_loader,
            cfg=cfg,
            train=True,
            optimizer=optimizer,
            epoch=epoch,
            accelerator=accelerator,
        )
        val_log = run_epoch(
            vla=vla,
            action_expert=action_expert,
            rtc_cfg=rtc_cfg,
            scheduler=scheduler,
            dataloader=val_loader,
            cfg=cfg,
            train=False,
            optimizer=None,
            epoch=epoch,
            accelerator=accelerator,
        )

        metrics: Dict[str, float] = {**train_log, **val_log}
        metrics["epoch"] = float(epoch)
        metrics["global_step"] = float(global_step)

        if accelerator.is_main_process:
            print(json.dumps(metrics, ensure_ascii=False))
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(metrics, ensure_ascii=False) + "\n")
            wandb.log(metrics, step=global_step)

        if accelerator.is_main_process and epoch % int(cfg.checkpoint.save_every) == 0:
            ep_path = os.path.join(out_dir, "checkpoints", f"epoch_{epoch:04d}.pt")
            save_checkpoint(
                ep_path,
                action_expert=action_expert,
                vla=vla,
                optimizer=optimizer,
                epoch=epoch,
                global_step=global_step,
                cfg=cfg,
                accelerator=accelerator,
            )
            save_checkpoint(
                os.path.join(out_dir, "checkpoints", "latest.pt"),
                action_expert=action_expert,
                vla=vla,
                optimizer=optimizer,
                epoch=epoch,
                global_step=global_step,
                cfg=cfg,
                accelerator=accelerator,
            )

        if accelerator.is_main_process and best_key in metrics:
            cur = float(metrics[best_key])
            better = cur < best_val if str(cfg.checkpoint.best_mode) == "min" else cur > best_val
            if better:
                best_val = cur
                save_checkpoint(
                    os.path.join(out_dir, "checkpoints", "best.pt"),
                    action_expert=action_expert,
                    vla=vla,
                    optimizer=optimizer,
                    epoch=epoch,
                    global_step=global_step,
                    cfg=cfg,
                    accelerator=accelerator,
                )

        global_step += int(train_log["train/updates"])
        accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        wandb.finish()
if __name__ == "__main__":
    os.chdir(ROOT_DIR)
    main()
