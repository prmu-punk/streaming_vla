from __future__ import annotations

from dataclasses import dataclass, field
import pathlib
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from action_tokenizers import OATActionTokenizer
from qwen3_vl import Qwen3VLForConditionalGeneration, Qwen3VLProcessor
from qwen3_vl.stream_runner import Qwen3VLStreamRunner


@dataclass
class StreamConfig:
    state_interval_s: float = 0.0
    vision_interval_s: float = 0.0
    obs_same_token: str = "<obs_same>"
    max_context_len: Optional[int] = None


@dataclass
class VLAConfig:
    model_name_or_path: str
    state_dim: int
    oat_tokenizer_checkpoint: str
    device: Optional[str] = None
    stream: StreamConfig = field(default_factory=StreamConfig)


def _load_vla_config(config_path: str) -> VLAConfig:
    with open(config_path, "r", encoding="utf-8") as f:
        raw: Dict[str, Any] = yaml.safe_load(f) or {}

    stream_raw = raw.get("stream", {}) or {}
    stream_cfg = StreamConfig(
        state_interval_s=float(stream_raw.get("state_interval_s", 0.0)),
        vision_interval_s=float(stream_raw.get("vision_interval_s", 0.0)),
        obs_same_token=str(stream_raw.get("obs_same_token", "<obs_same>")),
        max_context_len=stream_raw.get("max_context_len", None),
    )

    return VLAConfig(
        model_name_or_path=str(raw["model_name_or_path"]),
        state_dim=int(raw["state_dim"]),
        oat_tokenizer_checkpoint=str(raw["oat_tokenizer_checkpoint"]),
        device=raw.get("device", None),
        stream=stream_cfg,
    )


class Qwen3VLA(nn.Module):
    def __init__(
        self,
        config_path: Optional[str] = None,
        *,
        action_tokenizer: Optional[OATActionTokenizer] = None,
    ) -> None:
        super().__init__()
        if config_path is None:
            config_path = str(pathlib.Path(__file__).parent / "configs" / "vla_qwen3.yaml")
        cfg = _load_vla_config(config_path)

        device = cfg.device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.processor = Qwen3VLProcessor.from_pretrained(cfg.model_name_or_path, trust_remote_code=False)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(cfg.model_name_or_path, trust_remote_code=False)
        self.model.to(self.device)

        hidden_size = self.model.config.text_config.hidden_size
        self.state_encoder = nn.Sequential(
            nn.Linear(cfg.state_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

        if action_tokenizer is None:
            action_tokenizer = OATActionTokenizer(checkpoint=cfg.oat_tokenizer_checkpoint)
        self.action_tokenizer = action_tokenizer
        self.action_tokenizer.add_tokens(self.processor.tokenizer, self.model)

        self.stream_cfg = cfg.stream

    def new_runner(self) -> Qwen3VLStreamRunner:
        return Qwen3VLStreamRunner(
            model=self.model,
            state_interval_s=self.stream_cfg.state_interval_s,
            vision_interval_s=self.stream_cfg.vision_interval_s,
            state_encoder=self.state_encoder,
            state_token_id=0,
            max_context_len=self.stream_cfg.max_context_len,
            tokenizer=self.processor.tokenizer,
            obs_same_token=self.stream_cfg.obs_same_token,
        )

    def prefill(self, runner: Qwen3VLStreamRunner, prompt: Optional[str] = None) -> None:
        if prompt is None:
            eos_id = getattr(self.processor.tokenizer, "eos_token_id", None)
            if eos_id is None:
                raise ValueError("tokenizer.eos_token_id is required for prefill.")
            input_ids = torch.tensor([[eos_id]], device=self.device, dtype=torch.long)
        else:
            encoded = self.processor.tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
            input_ids = encoded["input_ids"].to(self.device)
        runner.prefill_text(input_ids=input_ids)

    def _make_video_tensor(
        self, frames: np.ndarray | torch.Tensor, num_frames: int
    ) -> torch.Tensor:
        if isinstance(frames, np.ndarray):
            frames_t = torch.from_numpy(frames)
        else:
            frames_t = frames
        if frames_t.dim() == 3:
            frames_t = frames_t.unsqueeze(0)
        if frames_t.shape[-1] == 3:
            frames_t = frames_t.permute(0, 3, 1, 2)
        if frames_t.shape[0] < num_frames:
            repeat = num_frames - frames_t.shape[0]
            frames_t = torch.cat([frames_t, frames_t[-1:].repeat(repeat, 1, 1, 1)], dim=0)
        elif frames_t.shape[0] > num_frames:
            frames_t = frames_t[:num_frames]
        return frames_t

    def insert_step(
        self,
        runner: Qwen3VLStreamRunner,
        frames: np.ndarray | torch.Tensor,
        *,
        state: torch.Tensor,
        ts: Optional[int] = None,
        num_frames: int = 4,
    ) -> bool:
        video = self._make_video_tensor(frames, num_frames)
        state_tokens = state.to(self.device)
        return runner.insert_step(
            processor=self.processor,
            video=video,
            state_tokens=state_tokens,
            ts=str(ts) if ts is not None else None,
        )

    def action_tokens(self, actions: torch.Tensor) -> torch.LongTensor:
        return self.action_tokenizer.tokenize(actions)

    def append_action_tokens_and_loss(
        self,
        runner: Qwen3VLStreamRunner,
        action_tokens: torch.LongTensor,
    ) -> torch.Tensor:
        action_tokens = action_tokens.to(self.device)
        logits = runner.append_text_tokens_with_logits(input_ids=action_tokens)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), action_tokens.reshape(-1))
        return loss

    def _sample_masked_next_token(
        self,
        logits: torch.Tensor,
        *,
        allowed_token_ids: torch.LongTensor,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.LongTensor:
        masked = torch.full_like(logits, -float("inf"))
        masked[:, allowed_token_ids] = logits[:, allowed_token_ids]

        if temperature <= 0:
            return torch.argmax(masked, dim=-1, keepdim=True)

        masked = masked / max(temperature, 1e-6)
        if top_k is not None and top_k > 0:
            k = min(int(top_k), masked.shape[-1])
            v, _ = torch.topk(masked, k=k, dim=-1)
            cutoff = v[:, [-1]]
            masked = torch.where(masked < cutoff, torch.full_like(masked, -float("inf")), masked)

        probs = torch.softmax(masked, dim=-1)
        return torch.multinomial(probs, num_samples=1)

    def generate_action_chunk(
        self,
        runner: Qwen3VLStreamRunner,
        *,
        fixed_action_tokens: int = 5,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """
        Generate exactly `fixed_action_tokens` action tokens after <act_bos>,
        then append <act_eos>. The generated action tokens are detokenized by OAT.
        Requires runner cache to already end with <act_bos>.
        """
        logits = runner.get_last_logits()
        if logits.shape[0] != 1:
            raise ValueError(f"Only batch size 1 is supported in action generation, got {logits.shape[0]}.")
        if fixed_action_tokens <= 0:
            raise ValueError(f"fixed_action_tokens must be positive, got {fixed_action_tokens}.")

        allowed = self.action_tokenizer.allowed_hf_token_ids(
            device=logits.device,
            include_eos=False,
        )
        eos_id = self.action_tokenizer.act_eos_hf_id

        generated: list[torch.LongTensor] = []
        for _ in range(fixed_action_tokens):
            next_token = self._sample_masked_next_token(
                logits,
                allowed_token_ids=allowed,
                temperature=temperature,
                top_k=top_k,
            )  # [1, 1]
            generated.append(next_token)
            logits = runner.generate_next(next_token)

        # Close action span with <act_eos>.
        eos_token = torch.tensor([[eos_id]], dtype=torch.long, device=logits.device)
        runner.append_text_tokens(input_ids=eos_token)

        action_token_ids = torch.cat(generated, dim=1)  # [1, fixed_action_tokens]
        action_chunk = self.action_tokenizer.detokenize(action_token_ids)

        return {
            "action_token_ids": action_token_ids,
            "action_chunk": action_chunk,
            "ended_by_eos": torch.tensor([True], device=self.device),
        }
