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
from accelerate import Accelerator, DistributedDataParallelKwargs
from multiprocessing.reduction import ForkingPickler
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, dataloader
from tqdm.auto import tqdm
from transformers import get_scheduler
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
from model.rtc_async.qwen3_stream.kv_export import export_selected_kv_cache
from model.rtc_async.training.loss_rtc import build_rtc_inpainting_batch, rtc_velocity_loss
from model.vla_qwen3_rtc import Qwen3RTCVLAEncoder
from normalization import RTCNormalizer


@dataclass
class RTCAsyncResolvedConfig:
    max_context_len: int | None
    selected_layers: list[int]
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
        raw = yaml.safe_load(f)

    stream = raw["stream"]
    action_expert = raw["action_expert"]

    return RTCAsyncResolvedConfig(
        max_context_len=int(float(stream["max_context_len"])),
        selected_layers=[int(x) for x in stream["selected_layers"]],
        action_expert=action_expert,
    )


def build_action_expert_runner(
    *,
    rtc_cfg: RTCAsyncResolvedConfig,
    state_dim: int,
    action_dim: int,
    horizon: int,
    cond_dim: int,
) -> ActionExpertRunner:
    action_cfg = dict(rtc_cfg.action_expert)
    runner_cfg = ActionExpertRunnerConfig(
        state_dim=state_dim,
        action_dim=int(action_cfg.pop("action_dim")),
        horizon=int(action_cfg.pop("horizon")),
        cond_dim=int(cond_dim),
        **action_cfg,
    )
    if runner_cfg.horizon != horizon:
        raise ValueError(f"horizon mismatch: config={runner_cfg.horizon}, resolved={horizon}")
    if runner_cfg.action_dim != action_dim:
        raise ValueError(f"action_dim mismatch: config={runner_cfg.action_dim}, dataset={action_dim}")
    return ActionExpertRunner(runner_cfg)


def _stack_anchor_state(sample_list: List[Dict[str, Any]], device: str) -> torch.Tensor:
    state = torch.stack([sample["anchor_state"] for sample in sample_list], dim=0)
    return state.to(device)


def run_epoch(
    *,
    vla: Qwen3RTCVLAEncoder,
    action_expert: ActionExpertRunner,
    rtc_cfg: RTCAsyncResolvedConfig,
    dataloader: DataLoader,
    cfg: DictConfig,
    train: bool,
    optimizer: torch.optim.Optimizer | None,
    scheduler: Any | None,
    epoch: int,
    accelerator: Accelerator,
) -> Dict[str, float]:
    train_vla = bool(cfg.rtc_async.train_vla)
    if train:
        action_expert.train()
        vla.train(train_vla)
    else:
        action_expert.eval()
        vla.eval()

    total_loss = 0.0
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
                source_dt_ms=int(cfg.training.source_dt_ms),
                return_condition_cache=True,
            )
        target_chunk = out["target_chunk"]
        past_key_values = out.get("past_key_values", None)
        attention_mask = out.get("attention_mask", None)
        prompt_mask = out.get("prompt_mask", None)
        step_mask = out.get("step_mask", None)
        if target_chunk is None or past_key_values is None:
            continue

        kv_cache, attention_mask, prompt_mask, step_mask = export_selected_kv_cache(
            past_key_values=past_key_values,
            selected_layers=rtc_cfg.selected_layers,
            clone=False,
            prompt_mask=prompt_mask,
            step_mask=step_mask,
        )

        state = _stack_anchor_state(sample_list, accelerator.device)
        zero_delay = torch.zeros(
            (int(target_chunk.shape[0]),),
            device=accelerator.device,
            dtype=torch.long,
        )
        full_chunk_batch = build_rtc_inpainting_batch(
            action=target_chunk,
            delay=zero_delay,
        )

        if not bool(cfg.rtc_async.use_attention_mask):
            attention_mask = None

        pred_u_t = action_expert(
            noisy_action=full_chunk_batch.x_t,
            state=state,
            time=full_chunk_batch.time,
            kv_cache=kv_cache,
            attention_mask=attention_mask,
            prompt_mask=prompt_mask,
            step_mask=step_mask,
        )
        loss = rtc_velocity_loss(pred_u_t=pred_u_t, batch=full_chunk_batch)

        if train:
            accelerator.backward(loss)
            if cfg.training.max_grad_norm is not None and accelerator.sync_gradients:
                params = list(action_expert.parameters())
                if train_vla:
                    params += list(vla.parameters())
                accelerator.clip_grad_norm_(params, float(cfg.training.max_grad_norm))
            assert optimizer is not None
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            n_updates += 1

        bs = int(target_chunk.shape[0])
        total_loss += float(loss.detach().item()) * bs
        n_samples += bs

        if accelerator.is_local_main_process and hasattr(progress, "set_postfix"):
            progress.set_postfix(
                full_loss=f"{(total_loss / max(n_samples, 1)):.4f}",
                samples=n_samples,
            )

        if max_batches is not None and (batch_idx + 1) >= int(max_batches):
            break

    local_stats = torch.tensor(
        [[total_loss, float(n_samples), float(n_updates)]],
        device=accelerator.device,
        dtype=torch.float64,
    )
    gathered_stats = accelerator.gather(local_stats)
    summed_stats = gathered_stats[:, :2].sum(dim=0)
    max_updates = gathered_stats[:, 2].max()

    prefix = "train" if train else "val"
    return {
        f"{prefix}/full_chunk_loss": float(summed_stats[0].item() / max(summed_stats[1].item(), 1.0)),
        f"{prefix}/samples": float(summed_stats[1].item()),
        f"{prefix}/updates": float(max_updates.item()),
    }


def save_checkpoint(
    path: str,
    *,
    action_expert: ActionExpertRunner,
    vla: Qwen3RTCVLAEncoder,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    normalizer: RTCNormalizer,
    epoch: int,
    global_step: int,
    cfg: DictConfig,
    accelerator: Accelerator,
) -> None:
    payload = {
        "epoch": int(epoch),
        "global_step": int(global_step),
        "action_expert": accelerator.get_state_dict(action_expert),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "normalization": normalizer.to_payload(),
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
    scheduler: Any | None = None,
    train_vla: bool,
    load_optimizer_state: bool = False,
    load_scheduler_state: bool = False,
) -> tuple[int, int]:
    payload = torch.load(path, map_location=vla.device)
    action_expert.load_state_dict(payload["action_expert"], strict=True)
    if "vla" in payload:
        vla.load_state_dict(payload["vla"], strict=True)
    if load_optimizer_state and optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    if load_scheduler_state and scheduler is not None and "scheduler" in payload:
        scheduler.load_state_dict(payload["scheduler"])
    epoch = int(payload.get("epoch", -1))
    global_step = int(payload.get("global_step", 0))
    return epoch, global_step


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent / "configs"),
    config_name="train_libero90_async",
)
def main(cfg: DictConfig) -> None:
    OmegaConf.resolve(cfg)
    ddp_kwargs = DistributedDataParallelKwargs(
        find_unused_parameters=False,
        static_graph=bool(cfg.rtc_async.train_vla),
    )
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
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

    chunk_horizon = int(rtc_cfg.action_expert["horizon"])

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
        chunk_horizon=int(chunk_horizon),
        anchor_stride_steps=anchor_stride,
        max_context_len=int(rtc_cfg.max_context_len),
        episode_cache_size=int(cfg.dataset.get("episode_cache_size", 8)),
        max_episodes=cfg.dataset.max_episodes,
        processor=vla.processor,
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

    action_mean, action_std = full_set.base.compute_action_stats(train_eps)
    state_mean, state_std = full_set.base.compute_state_stats(train_eps)
    normalizer = RTCNormalizer.from_stats(
        action_mean=action_mean,
        action_std=action_std,
        state_mean=state_mean,
        state_std=state_std,
    )
    full_set.set_normalization_stats(
        action_mean=action_mean,
        action_std=action_std,
        state_mean=state_mean,
        state_std=state_std,
    )

    train_indices = full_set.sample_indices_for_episodes(train_eps)
    val_indices = full_set.sample_indices_for_episodes(val_eps)

    if int(vla.state_dim) != full_set.state_dim:
        raise ValueError(
            f"state_dim mismatch: VLA expects {vla.state_dim}, dataset provides {full_set.state_dim}."
        )

    action_expert = build_action_expert_runner(
        rtc_cfg=rtc_cfg,
        state_dim=full_set.state_dim,
        action_dim=full_set.action_dim,
        horizon=full_set.chunk_horizon,
        cond_dim=int(vla.kv_cache_dim),
    ).to(accelerator.device)

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

    param_groups = [
        {
            "params": [p for p in action_expert.parameters() if p.requires_grad],
            "lr": float(cfg.optimizer.action_expert_lr),
        }
    ]
    if bool(cfg.rtc_async.train_vla):
        vla_params = [p for p in vla.parameters() if p.requires_grad]
        if vla_params:
            param_groups.append(
                {
                    "params": vla_params,
                    "lr": float(cfg.optimizer.vla_lr),
                }
            )

    optimizer = torch.optim.AdamW(
        param_groups,
        betas=(float(cfg.optimizer.beta1), float(cfg.optimizer.beta2)),
        weight_decay=float(cfg.optimizer.weight_decay),
    )
    num_update_steps_per_epoch = len(train_loader)
    max_train_steps = int(cfg.training.num_epochs) * int(num_update_steps_per_epoch)
    scheduler = get_scheduler(
        str(cfg.scheduler.name),
        optimizer=optimizer,
        num_warmup_steps=int(cfg.scheduler.warmup_steps),
        num_training_steps=max(max_train_steps, 1),
    )

    log_path = os.path.join(out_dir, "train_log.jsonl")
    best_key = "val/full_chunk_loss"
    best_val = math.inf

    global_step = 0
    start_epoch = 0
    resume_from = cfg.checkpoint.get("resume_from", None)
    if resume_from:
        resumed_epoch, global_step = load_checkpoint(
            str(resume_from),
            action_expert=action_expert,
            vla=vla,
            optimizer=optimizer,
            scheduler=scheduler,
            train_vla=bool(cfg.rtc_async.train_vla),
            load_optimizer_state=bool(cfg.checkpoint.get("load_optimizer_state", False)),
            load_scheduler_state=bool(cfg.checkpoint.get("load_scheduler_state", False)),
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
        vla, action_expert, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
            vla,
            action_expert,
            optimizer,
            train_loader,
            val_loader,
            scheduler,
        )
    else:
        action_expert, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
            action_expert,
            optimizer,
            train_loader,
            val_loader,
            scheduler,
        )

    for epoch in range(start_epoch, int(cfg.training.num_epochs)):
        train_bucket_sampler.set_epoch(epoch)
        train_log = run_epoch(
            vla=vla,
            action_expert=action_expert,
            rtc_cfg=rtc_cfg,
            dataloader=train_loader,
            cfg=cfg,
            train=True,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            accelerator=accelerator,
        )
        val_log = run_epoch(
            vla=vla,
            action_expert=action_expert,
            rtc_cfg=rtc_cfg,
            dataloader=val_loader,
            cfg=cfg,
            train=False,
            optimizer=None,
            scheduler=None,
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
                scheduler=scheduler,
                normalizer=normalizer,
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
                scheduler=scheduler,
                normalizer=normalizer,
                epoch=epoch,
                global_step=global_step,
                cfg=cfg,
                accelerator=accelerator,
            )

        if accelerator.is_main_process and best_key in metrics:
            cur = float(metrics[best_key])
            if cur < best_val:
                best_val = cur
                save_checkpoint(
                    os.path.join(out_dir, "checkpoints", "best.pt"),
                    action_expert=action_expert,
                    vla=vla,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    normalizer=normalizer,
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
