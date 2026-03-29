from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

KVCache = list[tuple[torch.Tensor, torch.Tensor]]

@dataclass
class ActionExpertConfig:
    state_dim: int
    action_dim: int
    horizon: int
    cond_dim: int
    hidden_size: int = 512
    num_layers: int = 8
    num_heads: int = 8
    mlp_ratio: float = 4.0
    norm_eps: float = 1e-6
    ffn_multiple_of: int = 256
    ffn_dim_multiplier: float | None = None

def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    if shift.dim() == 2:
        shift = shift.unsqueeze(1)
    if scale.dim() == 2:
        scale = scale.unsqueeze(1)
    return x * (1 + scale) + shift

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_float = x.float()
        rms = torch.rsqrt(x_float.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        y = x_float * rms
        return (y.to(dtype=x.dtype)) * self.weight

class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, freq_size: int = 256) -> None:
        super().__init__()
        self.freq_size = freq_size
        self.mlp = nn.Sequential(
            nn.Linear(freq_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def _sinusoidal_embedding(self, t: torch.Tensor) -> torch.Tensor:
        half = self.freq_size // 2
        device = t.device
        dtype = t.dtype
        freq = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=device, dtype=dtype) / max(half - 1, 1)
        )
        flat_t = t.reshape(-1)
        args = flat_t[:, None] * freq[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.freq_size % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb.reshape(*t.shape, self.freq_size)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.dim() not in (1, 2):
            raise ValueError(f"time must be [B] or [B,H], got {tuple(t.shape)}")
        return self.mlp(self._sinusoidal_embedding(t))

class FeedForward(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: float | None,
    ) -> None:
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class DiTBlock(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float,
        norm_eps: float,
        ffn_multiple_of: int,
        ffn_dim_multiplier: float | None,
    ) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(hidden_size, eps=norm_eps)
        self.cross_norm = RMSNorm(hidden_size, eps=norm_eps)
        self.cond_norm = RMSNorm(hidden_size, eps=norm_eps)
        self.ffn_norm = RMSNorm(hidden_size, eps=norm_eps)

        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.ffn = FeedForward(
            dim=hidden_size,
            hidden_dim=int(hidden_size * mlp_ratio),
            multiple_of=ffn_multiple_of,
            ffn_dim_multiplier=ffn_dim_multiplier,
        )

        self.ada = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size * 2, hidden_size * 9, bias=True),
        )
        ada_linear = self.ada[1]
        if not isinstance(ada_linear, nn.Linear):
            raise TypeError("DiTBlock.ada[1] must be nn.Linear")
        nn.init.zeros_(ada_linear.weight)
        nn.init.zeros_(ada_linear.bias)

    def forward(
        self,
        x: torch.Tensor,
        *,
        ada_cond: torch.Tensor,
        ck: Optional[torch.Tensor] = None,
        cv: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        (
            shift_attn,
            scale_attn,
            gate_attn,
            shift_cross,
            scale_cross,
            gate_cross,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = self.ada(ada_cond).chunk(9, dim=-1)

        attn_in = _modulate(self.attn_norm(x), shift_attn, scale_attn)
        attn_out, _ = self.attn(attn_in, attn_in, attn_in, need_weights=False)
        h = x + self._expand_gate(gate_attn) * attn_out

        if ck is not None and cv is not None:
            q = _modulate(self.cross_norm(h), shift_cross, scale_cross)
            k = self.cond_norm(ck)
            v = self.cond_norm(cv)
            cross_out, _ = self.cross_attn(q, k, v, key_padding_mask=attn_mask, need_weights=False)
            h = h + self._expand_gate(gate_cross) * cross_out

        mlp_out = self.ffn(_modulate(self.ffn_norm(h), shift_mlp, scale_mlp))
        return h + self._expand_gate(gate_mlp) * mlp_out

    @staticmethod
    def _expand_gate(gate: torch.Tensor) -> torch.Tensor:
        if gate.dim() == 2:
            return gate.unsqueeze(1)
        if gate.dim() == 3:
            return gate
        raise ValueError(f"gate must be [B,C] or [B,H,C], got {tuple(gate.shape)}")

class ActionExpertBackbone(nn.Module):
    def __init__(self, config: ActionExpertConfig) -> None:
        super().__init__()
        self.config = config
        self.action_in = nn.Linear(config.action_dim, config.hidden_size)
        self.state_global_in = nn.Linear(config.state_dim, config.hidden_size)
        self.time_embed = TimestepEmbedder(config.hidden_size)
        self.prompt_k_proj = nn.Linear(config.cond_dim, config.hidden_size)
        self.prompt_v_proj = nn.Linear(config.cond_dim, config.hidden_size)
        self.step_k_proj = nn.Linear(config.cond_dim, config.hidden_size)
        self.step_v_proj = nn.Linear(config.cond_dim, config.hidden_size)
        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    hidden_size=config.hidden_size,
                    num_heads=config.num_heads,
                    mlp_ratio=config.mlp_ratio,
                    norm_eps=config.norm_eps,
                    ffn_multiple_of=config.ffn_multiple_of,
                    ffn_dim_multiplier=config.ffn_dim_multiplier,
                )
                for _ in range(config.num_layers)
            ]
        )
        self.final_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.final_ada = nn.Sequential(
            nn.SiLU(),
            nn.Linear(config.hidden_size * 2, config.hidden_size * 2, bias=True),
        )
        final_ada_linear = self.final_ada[1]
        if not isinstance(final_ada_linear, nn.Linear):
            raise TypeError("ActionExpertBackbone.final_ada[1] must be nn.Linear")
        nn.init.zeros_(final_ada_linear.weight)
        nn.init.zeros_(final_ada_linear.bias)
        self.out_proj = nn.Linear(config.hidden_size, config.action_dim)

    @staticmethod
    def _to_horizon_state(state: torch.Tensor, batch_size: int, horizon: int) -> torch.Tensor:
        if state.dim() == 2:
            state = state.unsqueeze(1)
        if state.shape[1] == 1:
            state = state.expand(batch_size, horizon, -1)
        if state.shape[1] != horizon:
            raise ValueError(f"state horizon mismatch: got {state.shape[1]}, expected {horizon}")
        return state

    @staticmethod
    def _kv_to_tokens(x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:

            if x.shape[1] < x.shape[2]:
                x = x.permute(0, 2, 1, 3)

            x = x.reshape(x.shape[0], x.shape[1], -1)
            return x
        if x.dim() == 3:
            return x
        if x.dim() == 2:
            return x.unsqueeze(1)
        raise ValueError(f"Unsupported KV rank: {x.dim()}, shape={tuple(x.shape)}")

    def _encode_layerwise_kv(
        self,
        kv_cache: KVCache | None,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        attention_mask: Optional[torch.Tensor] = None,
        prompt_mask: Optional[torch.Tensor] = None,
        step_mask: Optional[torch.Tensor] = None,
    ) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        if not kv_cache:
            return []
        layer_conds: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        for k, v in kv_cache:
            if k.shape[0] != batch_size or v.shape[0] != batch_size:
                raise ValueError(
                    f"kv cache batch mismatch: k={tuple(k.shape)}, v={tuple(v.shape)}, expected B={batch_size}"
                )
            k_tokens = self._kv_to_tokens(k.to(device=device, dtype=dtype))
            v_tokens = self._kv_to_tokens(v.to(device=device, dtype=dtype))
            kv_len = int(k_tokens.shape[1])
            valid_mask = self._align_condition_mask(
                mask=attention_mask,
                kv_len=kv_len,
                device=device,
                default=True,
            )
            prompt_valid = self._align_condition_mask(
                mask=prompt_mask,
                kv_len=kv_len,
                device=device,
                default=False,
            )
            step_valid = self._align_condition_mask(
                mask=step_mask,
                kv_len=kv_len,
                device=device,
                default=False,
            )
            if prompt_mask is not None or step_mask is not None:
                cond_mask = valid_mask & (prompt_valid | step_valid)
                k_proj = (
                    self.prompt_k_proj(k_tokens) * prompt_valid.unsqueeze(-1).to(dtype)
                    + self.step_k_proj(k_tokens) * step_valid.unsqueeze(-1).to(dtype)
                )
                v_proj = (
                    self.prompt_v_proj(v_tokens) * prompt_valid.unsqueeze(-1).to(dtype)
                    + self.step_v_proj(v_tokens) * step_valid.unsqueeze(-1).to(dtype)
                )
            else:
                cond_mask = valid_mask
                k_proj = self.prompt_k_proj(k_tokens)
                v_proj = self.prompt_v_proj(v_tokens)
            layer_conds.append((k_proj, v_proj, cond_mask))
        return layer_conds

    def forward(
        self,
        *,
        noisy_action: torch.Tensor,
        state: torch.Tensor,
        time: torch.Tensor,
        kv_cache: KVCache | None = None,
        attention_mask: Optional[torch.Tensor] = None,
        prompt_mask: Optional[torch.Tensor] = None,
        step_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        batch_size, horizon, _ = noisy_action.shape
        state = self._to_horizon_state(state, batch_size, horizon)
        if time.dim() == 1:
            time = time[:, None].expand(batch_size, horizon)
        elif time.dim() == 2:
            if time.shape[0] != batch_size or time.shape[1] != horizon:
                raise ValueError(f"time must be [B] or [B,H], got {tuple(time.shape)} expected {(batch_size, horizon)}")
        else:
            raise ValueError(f"time must be [B] or [B,H], got {tuple(time.shape)}")
        if attention_mask is not None and attention_mask.dim() != 2:
            raise ValueError(f"attention_mask must be [B, S], got {tuple(attention_mask.shape)}")
        if attention_mask is not None and attention_mask.shape[0] != batch_size:
            raise ValueError(
                f"attention_mask batch mismatch: got {attention_mask.shape[0]}, expected {batch_size}"
            )

        x = self.action_in(noisy_action)
        time_cond = self.time_embed(time)
        state_cond = self.state_global_in(state.mean(dim=1))[:, None, :].expand(batch_size, horizon, -1)
        ada_cond = torch.cat([time_cond, state_cond], dim=-1)

        kv_layer_conds = self._encode_layerwise_kv(
            kv_cache,
            batch_size=batch_size,
            device=noisy_action.device,
            dtype=noisy_action.dtype,
            attention_mask=attention_mask,
            prompt_mask=prompt_mask,
            step_mask=step_mask,
        )

        for i, block in enumerate(self.blocks):
            if kv_layer_conds:
                ck, cv, cond_mask = kv_layer_conds[i % len(kv_layer_conds)]
                key_padding_mask = ~cond_mask
            else:
                ck = None
                cv = None
                key_padding_mask = None
            x = block(
                x,
                ada_cond=ada_cond,
                ck=ck,
                cv=cv,
                attn_mask=key_padding_mask,
            )

        shift, scale = self.final_ada(ada_cond).chunk(2, dim=-1)
        x = _modulate(self.final_norm(x), shift, scale)
        return self.out_proj(x)

    @staticmethod
    def _align_condition_mask(
        *,
        mask: Optional[torch.Tensor],
        kv_len: int,
        device: torch.device,
        default: bool,
    ) -> torch.Tensor:
        if mask is None:
            return torch.full((1, kv_len), bool(default), device=device, dtype=torch.bool)

        valid_mask = mask.to(device=device)
        if valid_mask.dtype != torch.bool:
            valid_mask = valid_mask > 0

        seq_len = valid_mask.shape[1]
        if seq_len > kv_len:
            valid_mask = valid_mask[:, -kv_len:]
        elif seq_len < kv_len:
            pad_value = bool(default)
            pad = torch.full((valid_mask.shape[0], kv_len - seq_len), pad_value, device=device, dtype=torch.bool)
            valid_mask = torch.cat([pad, valid_mask], dim=1)
        return valid_mask
