import json
import math
import os
import pathlib
import random
import sys
from typing import Dict, List, Optional

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, random_split

ROOT_DIR = str(pathlib.Path(__file__).parent.parent)
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from libero90_dataset import LiberoChunkDataset
from vla_qwen3 import Qwen3VLA


# Optional import for rollout evaluation.
_OAT_ROOT = pathlib.Path(__file__).resolve().parents[2] / "lingbot-va" / "oat"
if str(_OAT_ROOT) not in sys.path:
    sys.path.append(str(_OAT_ROOT))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str) -> None:
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)


def build_video_from_image(image_t: torch.Tensor, num_frames: int) -> np.ndarray:
    """
    Convert a single image [H, W, 3] into a repeated pseudo-video [T, H, W, 3].
    """
    if image_t.dim() != 3:
        raise ValueError(f"image_t must be [H, W, 3], got {tuple(image_t.shape)}")
    image = image_t.detach().cpu().numpy()
    if image.shape[-1] != 3:
        raise ValueError(f"image_t channel dim must be 3, got shape {image.shape}")
    return np.repeat(image[None, ...], repeats=int(num_frames), axis=0)


def infer_chunk_horizon(vla: Qwen3VLA, fixed_action_tokens: int) -> int:
    allowed = vla.action_tokenizer.allowed_hf_token_ids(device=vla.device, include_eos=False)
    if allowed.numel() == 0:
        raise ValueError("No allowed action token ids found.")
    probe_id = allowed[0].view(1, 1).repeat(1, fixed_action_tokens)
    with torch.no_grad():
        chunk = vla.action_tokenizer.detokenize(probe_id)
    if chunk.dim() != 3:
        raise ValueError(f"detokenized chunk must be [B, H, D], got {tuple(chunk.shape)}")
    return int(chunk.shape[1])


def rollout_eval_sync(vla: Qwen3VLA, cfg: DictConfig) -> Dict[str, float]:
    from oat.env.libero.env import LiberoEnv  # type: ignore
    from oat.env.libero.factory import get_subtasks  # type: ignore

    rollout_cfg = cfg.eval.rollout
    if int(rollout_cfg.max_tasks) <= 0 or int(rollout_cfg.n_eval_per_task) <= 0:
        return {}

    vla.eval()
    subtasks = get_subtasks(str(rollout_cfg.task_name))[: int(rollout_cfg.max_tasks)]
    task_success: Dict[str, float] = {}
    all_success: List[float] = []

    with torch.no_grad():
        for task_name in subtasks:
            successes: List[float] = []
            for seed_idx in range(int(rollout_cfg.n_eval_per_task)):
                env = LiberoEnv(
                    task_name=task_name,
                    image_size=int(rollout_cfg.image_size),
                    seed=int(rollout_cfg.seed_base) + seed_idx,
                    camera_names=[str(rollout_cfg.camera_name)],
                    state_ports=[str(k) for k in cfg.dataset.state_keys],
                    max_episode_steps=int(rollout_cfg.max_episode_steps),
                    enable_render=bool(rollout_cfg.enable_render),
                )

                obs, _ = env.reset()
                runner = vla.new_runner()
                vla.prefill(runner, prompt=str(obs.get("prompt", "")))

                episode_success = False
                steps = 0
                while steps < int(rollout_cfg.max_episode_steps):
                    steps += 1
                    image = torch.from_numpy(obs[str(rollout_cfg.camera_name) + "_rgb"])
                    video = build_video_from_image(image, num_frames=int(cfg.model.num_frames))

                    state_np = np.concatenate(
                        [np.asarray(obs[k], dtype=np.float32).reshape(-1) for k in cfg.dataset.state_keys],
                        axis=0,
                    )
                    state_t = torch.from_numpy(state_np).to(vla.device).unsqueeze(0)

                    inserted = vla.insert_step(
                        runner,
                        frames=video,
                        state=state_t,
                        ts=steps,
                        num_frames=int(cfg.model.num_frames),
                    )
                    if not inserted:
                        continue

                    out = vla.generate_action_chunk(
                        runner,
                        fixed_action_tokens=int(cfg.model.fixed_action_tokens),
                        temperature=float(rollout_cfg.temperature),
                        top_k=None,
                    )
                    action_chunk = out["action_chunk"][0].detach().cpu().numpy()

                    done = False
                    for i in range(action_chunk.shape[0]):
                        obs, reward, done, _, _ = env.step(action_chunk[i])
                        if float(reward) >= 1.0:
                            episode_success = True
                        if done:
                            break
                    if done:
                        break

                env.close()
                successes.append(1.0 if episode_success else 0.0)
                all_success.append(successes[-1])

            task_success[f"rollout/{task_name}_success"] = float(np.mean(successes))

    result: Dict[str, float] = dict(task_success)
    result["rollout/mean_success_rate"] = float(np.mean(all_success)) if all_success else 0.0
    return result


def train_one_epoch(
    vla: Qwen3VLA,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    cfg: DictConfig,
) -> Dict[str, float]:
    vla.train()
    fixed_tokens = int(cfg.model.fixed_action_tokens)

    total_loss = 0.0
    total_mse = 0.0
    n_batches = 0

    for batch in dataloader:
        image_t = batch["image_t"]
        state_t = batch["state_t"]
        gt_chunk = batch["gt_chunk"].to(vla.device)
        valid_len = int(batch["valid_len"][0].item())

        if image_t.shape[0] != 1:
            raise ValueError(
                f"This first sync version only supports batch_size=1, got {image_t.shape[0]}"
            )

        # Build one-step context then teacher-force fixed K action tokens.
        runner = vla.new_runner()
        vla.prefill(runner, prompt=None)

        video = build_video_from_image(image_t[0], num_frames=int(cfg.model.num_frames))
        inserted = vla.insert_step(
            runner,
            frames=video,
            state=state_t.to(vla.device),
            ts=int(batch["start_t"][0].item()),
            num_frames=int(cfg.model.num_frames),
        )
        if not inserted:
            continue

        gt_valid_chunk = gt_chunk[:, :valid_len, :]
        gt_tokens = vla.action_tokens(gt_valid_chunk)
        token_len = int(gt_tokens.shape[1])
        if token_len > fixed_tokens:
            raise ValueError(
                f"gt token length {token_len} exceeds fixed_action_tokens={fixed_tokens}."
            )

        logits = runner.append_text_tokens_with_logits(input_ids=gt_tokens)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), gt_tokens.reshape(-1))

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.training.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(vla.parameters(), float(cfg.training.max_grad_norm))
        optimizer.step()

        with torch.no_grad():
            pred = vla.action_tokenizer.detokenize(gt_tokens)
            mse = F.mse_loss(pred[:, :valid_len, :], gt_valid_chunk)

        total_loss += float(loss.item())
        total_mse += float(mse.item())
        n_batches += 1

        if cfg.training.max_train_steps is not None and n_batches >= int(cfg.training.max_train_steps):
            break

    denom = max(n_batches, 1)
    return {
        "train/token_ce": total_loss / denom,
        "train/action_mse": total_mse / denom,
    }


@torch.no_grad()
def validate(
    vla: Qwen3VLA,
    dataloader: DataLoader,
    cfg: DictConfig,
) -> Dict[str, float]:
    vla.eval()
    fixed_tokens = int(cfg.model.fixed_action_tokens)

    total_loss = 0.0
    total_mse = 0.0
    n_batches = 0

    for batch in dataloader:
        image_t = batch["image_t"]
        state_t = batch["state_t"]
        gt_chunk = batch["gt_chunk"].to(vla.device)
        valid_len = int(batch["valid_len"][0].item())

        if image_t.shape[0] != 1:
            raise ValueError("Validation currently supports batch_size=1 only.")

        runner = vla.new_runner()
        vla.prefill(runner, prompt=None)

        video = build_video_from_image(image_t[0], num_frames=int(cfg.model.num_frames))
        inserted = vla.insert_step(
            runner,
            frames=video,
            state=state_t.to(vla.device),
            ts=int(batch["start_t"][0].item()),
            num_frames=int(cfg.model.num_frames),
        )
        if not inserted:
            continue

        gt_valid_chunk = gt_chunk[:, :valid_len, :]
        gt_tokens = vla.action_tokens(gt_valid_chunk)
        token_len = int(gt_tokens.shape[1])
        if token_len > fixed_tokens:
            raise ValueError(
                f"gt token length {token_len} exceeds fixed_action_tokens={fixed_tokens}."
            )

        logits = runner.append_text_tokens_with_logits(input_ids=gt_tokens)
        ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), gt_tokens.reshape(-1))

        pred_out = vla.generate_action_chunk(
            runner,
            fixed_action_tokens=fixed_tokens,
            temperature=0.0,
            top_k=None,
        )
        pred_chunk = pred_out["action_chunk"]
        mse = F.mse_loss(pred_chunk[:, :valid_len, :], gt_valid_chunk)

        total_loss += float(ce.item())
        total_mse += float(mse.item())
        n_batches += 1

        if cfg.eval.max_val_steps is not None and n_batches >= int(cfg.eval.max_val_steps):
            break

    denom = max(n_batches, 1)
    return {
        "val/token_ce": total_loss / denom,
        "val/action_mse": total_mse / denom,
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


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent / "configs"),
    config_name="train_libero90_sync",
)
def main(cfg: DictConfig) -> None:
    OmegaConf.resolve(cfg)
    set_seed(int(cfg.training.seed))

    output_dir = os.getcwd()
    ensure_dir(output_dir)
    ensure_dir(os.path.join(output_dir, "checkpoints"))

    vla = Qwen3VLA(config_path=str(cfg.model.vla_config_path))

    inferred_horizon = infer_chunk_horizon(vla, fixed_action_tokens=int(cfg.model.fixed_action_tokens))
    configured_horizon = cfg.dataset.chunk_horizon
    if configured_horizon is None:
        chunk_horizon = inferred_horizon
    else:
        chunk_horizon = int(configured_horizon)
    if chunk_horizon != inferred_horizon:
        raise ValueError(
            f"Chunk horizon mismatch: cfg={chunk_horizon}, inferred={inferred_horizon}. "
            "train/infer must use same chunk horizon."
        )
    stride_cfg = cfg.dataset.stride
    stride = chunk_horizon if stride_cfg is None else int(stride_cfg)
    if stride != chunk_horizon:
        raise ValueError(
            f"Stride mismatch: stride={stride}, chunk_horizon={chunk_horizon}. "
            "This sync version requires stride == chunk_horizon."
        )

    dataset = LiberoChunkDataset(
        zarr_path=str(cfg.dataset.zarr_path),
        chunk_horizon=chunk_horizon,
        stride=stride,
        image_key=str(cfg.dataset.image_key),
        action_key=str(cfg.dataset.action_key),
        state_keys=[str(k) for k in cfg.dataset.state_keys],
        max_episodes=cfg.dataset.max_episodes,
    )

    if int(cfg.model.state_dim) != dataset.state_dim:
        raise ValueError(
            f"state_dim mismatch: model expects {cfg.model.state_dim}, dataset provides {dataset.state_dim}."
        )

    val_ratio = float(cfg.dataset.val_ratio)
    if not (0.0 < val_ratio < 1.0):
        raise ValueError(f"val_ratio must be in (0,1), got {val_ratio}")

    n_total = len(dataset)
    n_val = max(1, int(round(n_total * val_ratio)))
    n_train = max(1, n_total - n_val)
    if n_train + n_val != n_total:
        n_val = n_total - n_train

    train_set, val_set = random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(int(cfg.training.seed)),
    )

    train_loader = DataLoader(train_set, **cfg.dataloader)
    val_loader = DataLoader(val_set, **cfg.val_dataloader)

    optimizer = torch.optim.AdamW(
        vla.parameters(),
        lr=float(cfg.optimizer.lr),
        betas=(float(cfg.optimizer.beta1), float(cfg.optimizer.beta2)),
        weight_decay=float(cfg.optimizer.weight_decay),
    )

    log_path = os.path.join(output_dir, "train_log.jsonl")
    best_key = str(cfg.checkpoint.best_monitor)
    best_value = math.inf if str(cfg.checkpoint.best_mode) == "min" else -math.inf

    global_step = 0
    for epoch in range(int(cfg.training.num_epochs)):
        train_log = train_one_epoch(vla=vla, dataloader=train_loader, optimizer=optimizer, cfg=cfg)
        val_log = validate(vla=vla, dataloader=val_loader, cfg=cfg)

        metrics: Dict[str, float] = {**train_log, **val_log}

        if bool(cfg.eval.enable_rollout_eval) and (epoch % int(cfg.eval.rollout_every) == 0):
            rollout_log = rollout_eval_sync(vla=vla, cfg=cfg)
            metrics.update(rollout_log)

        metrics["epoch"] = float(epoch)
        metrics["global_step"] = float(global_step)
        print(json.dumps(metrics, ensure_ascii=False))

        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(metrics, ensure_ascii=False) + "\n")

        if epoch % int(cfg.checkpoint.save_every) == 0:
            ckpt_path = os.path.join(output_dir, "checkpoints", f"epoch_{epoch:04d}.pt")
            save_checkpoint(ckpt_path, vla, optimizer, epoch, global_step, cfg)
            latest_path = os.path.join(output_dir, "checkpoints", "latest.pt")
            save_checkpoint(latest_path, vla, optimizer, epoch, global_step, cfg)

        if best_key in metrics:
            cur = float(metrics[best_key])
            is_better = cur < best_value if str(cfg.checkpoint.best_mode) == "min" else cur > best_value
            if is_better:
                best_value = cur
                best_path = os.path.join(output_dir, "checkpoints", "best.pt")
                save_checkpoint(best_path, vla, optimizer, epoch, global_step, cfg)

        global_step += len(train_loader)


if __name__ == "__main__":
    os.chdir(ROOT_DIR)
    main()
