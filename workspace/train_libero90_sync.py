import json
import math
import os
import pathlib
import random
import sys
from typing import Dict

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, dataloader
from tqdm.auto import tqdm
from multiprocessing.reduction import ForkingPickler
import wandb
default_collate_func = dataloader.default_collate
def default_collate_override(batch):
  dataloader._use_shared_memory = False
  return default_collate_func(batch)

setattr(dataloader, 'default_collate', default_collate_override)

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

from dataset.libero90_offline_context_dataset import LiberoOfflineContextDataset, offline_context_collate
from model.vla_qwen3 import Qwen3VLA


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str) -> None:
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)


def infer_chunk_horizon(vla: Qwen3VLA, fixed_action_tokens: int) -> int:
    allowed = vla.action_tokenizer.allowed_hf_token_ids(device=vla.device, include_eos=False)
    if allowed.numel() == 0:
        raise ValueError("No allowed action token ids found.")
    probe = allowed[0].view(1, 1).repeat(1, fixed_action_tokens)
    with torch.no_grad():
        chunk = vla.action_tokenizer.detokenize(probe)
    if chunk.dim() != 3:
        raise ValueError(f"detokenized chunk must be [B, H, D], got {tuple(chunk.shape)}")
    return int(chunk.shape[1])


def run_epoch(
    *,
    vla: Qwen3VLA,
    dataloader: DataLoader,
    cfg: DictConfig,
    train: bool,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
) -> Dict[str, float]:
    if train:
        vla.train()
    else:
        vla.eval()

    total_ce = 0.0
    total_mse = 0.0
    n_samples = 0
    n_tokens = 0.0
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

        with torch.set_grad_enabled(train):
            out = vla.forward_offline_context_batch(
                samples=sample_list,
                fixed_action_tokens=int(cfg.model.fixed_action_tokens),
                num_frames=int(cfg.model.num_frames),
                source_dt_ms=int(cfg.training.source_dt_ms),
            )
            logits = out["logits"]
            tgt_tokens = out["target_tokens"]
            tgt_chunk = out["target_chunk"]
            n_valid_tokens = out["n_valid_tokens"]
            labels = out.get("labels", None)
            if (
                logits is None
                or tgt_tokens is None
                or tgt_chunk is None
                or n_valid_tokens is None
                or labels is None
            ):
                continue

            ce = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                labels.reshape(-1),
                ignore_index=-100,
            )

            if train:
                ce.backward()
                if cfg.training.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(vla.parameters(), float(cfg.training.max_grad_norm))
                optimizer.step()
                n_updates += 1

        with torch.no_grad():
            allowed_action_ids = vla.action_tokenizer.allowed_hf_token_ids(
                device=logits.device,
                include_eos=False,
            )
            masked_logits = torch.full_like(logits, -float("inf"))
            masked_logits[..., allowed_action_ids] = logits[..., allowed_action_ids]
            pred_tokens = torch.argmax(masked_logits, dim=-1)
            pred_target_tokens = []
            target_chunks = []
            for row_idx in range(pred_tokens.shape[0]):
                valid_mask = (labels[row_idx] != -100)
                row_tokens = pred_tokens[row_idx][valid_mask]
                if row_tokens.numel() == 0:
                    continue
                pred_target_tokens.append(row_tokens)
                target_chunks.append(tgt_chunk[row_idx])
            if pred_target_tokens:
                pred_target_tokens = torch.stack(pred_target_tokens, dim=0)
                target_chunks = torch.stack(target_chunks, dim=0)
                detok = vla.action_tokenizer.detokenize(pred_target_tokens)
                mse = torch.nn.functional.mse_loss(detok, target_chunks).item()
            else:
                mse = 0.0

            bs = int(tgt_chunk.shape[0])
            token_weight = float(n_valid_tokens.item())
            total_ce += float(ce.detach().item()) * token_weight
            total_mse += float(mse) * bs
            n_samples += bs
            n_tokens += token_weight
            progress.set_postfix(
                token_ce=f"{(total_ce / max(n_tokens, 1.0)):.4f}",
                action_mse=f"{(total_mse / max(n_samples, 1)):.4f}",
                samples=n_samples,
            )

        if max_batches is not None and (batch_idx + 1) >= int(max_batches):
            break

    sample_denom = max(n_samples, 1)
    token_denom = max(n_tokens, 1.0)
    prefix = "train" if train else "val"
    return {
        f"{prefix}/token_ce": total_ce / token_denom,
        f"{prefix}/action_mse": total_mse / sample_denom,
        f"{prefix}/samples": float(n_samples),
        f"{prefix}/tokens": float(n_tokens),
        f"{prefix}/updates": float(n_updates),
    }


def save_checkpoint(
    path: str,
    vla: Qwen3VLA,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    cfg: DictConfig,
) -> None:
    payload = {
        "epoch": int(epoch),
        "global_step": int(global_step),
        "model": vla.state_dict(),
        "optimizer": optimizer.state_dict(),
        "cfg": OmegaConf.to_container(cfg, resolve=True),
    }
    torch.save(payload, path)


def load_checkpoint(
    path: str,
    vla: Qwen3VLA,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[int, int]:
    payload = torch.load(path, map_location=vla.device)
    vla.load_state_dict(payload["model"], strict=True)
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    epoch = int(payload.get("epoch", -1))
    global_step = int(payload.get("global_step", 0))
    return epoch, global_step


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent / "configs"),
    config_name="train_libero90_sync",
)
def main(cfg: DictConfig) -> None:
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

    vla = Qwen3VLA(config_path=str(cfg.model.vla_config_path))
    chunk_horizon = infer_chunk_horizon(vla, fixed_action_tokens=int(cfg.model.fixed_action_tokens))

    if cfg.dataset.get("chunk_horizon", None) is not None and int(cfg.dataset.chunk_horizon) != int(chunk_horizon):
        raise ValueError(
            f"chunk_horizon mismatch: cfg={cfg.dataset.chunk_horizon}, inferred={chunk_horizon}."
        )

    anchor_stride = cfg.dataset.get("anchor_stride_steps", None)
    if anchor_stride is None:
        anchor_stride = 1
    anchor_stride = int(anchor_stride)

    base_ep = LiberoOfflineContextDataset(
        zarr_path=str(cfg.dataset.zarr_path),
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
        fixed_action_tokens=int(cfg.model.fixed_action_tokens),
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
        zarr_path=str(cfg.dataset.zarr_path),
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
        fixed_action_tokens=int(cfg.model.fixed_action_tokens),
        max_episodes=cfg.dataset.max_episodes,
        episode_indices=train_eps,
    )
    val_set = LiberoOfflineContextDataset(
        zarr_path=str(cfg.dataset.zarr_path),
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
        fixed_action_tokens=int(cfg.model.fixed_action_tokens),
        max_episodes=cfg.dataset.max_episodes,
        episode_indices=val_eps,
    )

    if int(cfg.model.state_dim) != train_set.state_dim:
        raise ValueError(
            f"state_dim mismatch: model expects {cfg.model.state_dim}, dataset provides {train_set.state_dim}."
        )

    train_loader = DataLoader(train_set, collate_fn=offline_context_collate, **cfg.dataloader)
    val_loader = DataLoader(val_set, collate_fn=offline_context_collate, **cfg.val_dataloader)

    optimizer = torch.optim.AdamW(
        vla.parameters(),
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
        resumed_epoch, global_step = load_checkpoint(str(resume_from), vla, optimizer)
        start_epoch = resumed_epoch + 1
        print(
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

    for epoch in range(start_epoch, int(cfg.training.num_epochs)):
        train_log = run_epoch(
            vla=vla,
            dataloader=train_loader,
            cfg=cfg,
            train=True,
            optimizer=optimizer,
            epoch=epoch,
        )
        val_log = run_epoch(
            vla=vla,
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
            save_checkpoint(ep_path, vla, optimizer, epoch, global_step, cfg)
            save_checkpoint(os.path.join(out_dir, "checkpoints", "latest.pt"), vla, optimizer, epoch, global_step, cfg)

        if best_key in metrics:
            cur = float(metrics[best_key])
            better = cur < best_val if str(cfg.checkpoint.best_mode) == "min" else cur > best_val
            if better:
                best_val = cur
                save_checkpoint(os.path.join(out_dir, "checkpoints", "best.pt"), vla, optimizer, epoch, global_step, cfg)

        global_step += int(train_log["train/updates"])

    wandb.finish()


if __name__ == "__main__":
    os.chdir(ROOT_DIR)
    main()
