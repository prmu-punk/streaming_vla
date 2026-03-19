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
    """动作专家主干网络结构配置。"""

    state_dim: int
    action_dim: int
    horizon: int
    hidden_size: int = 512
    num_layers: int = 8
    num_heads: int = 8
    mlp_ratio: float = 4.0
    time_embed_dim: int = 256
    norm_eps: float = 1e-6
    ffn_multiple_of: int = 256
    ffn_dim_multiplier: float | None = None


def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """执行 AdaLN 调制：`x * (1 + scale) + shift`，支持 `[B,D]` 自动扩展到 `[B,1,D]`。"""
    if shift.dim() == 2:
        shift = shift.unsqueeze(1)
    if scale.dim() == 2:
        scale = scale.unsqueeze(1)
    return x * (1 + scale) + shift


class RMSNorm(nn.Module):
    """RDT 对齐版 RMSNorm。"""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        """初始化 RMSNorm。

        参数:
            dim: 通道维度。
            eps: 数值稳定项。
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """对输入执行 RMS 归一化并应用可学习缩放。"""
        x_float = x.float()
        rms = torch.rsqrt(x_float.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        y = x_float * rms
        return (y.to(dtype=x.dtype)) * self.weight


class TimestepEmbedder(nn.Module):
    """将标量扩散时间映射到隐藏向量。"""

    def __init__(self, hidden_size: int, freq_size: int = 256) -> None:
        """初始化扩散时间嵌入器。

        参数:
            hidden_size: 输出隐藏维度。
            freq_size: 正弦位置编码频率维度。
        """
        super().__init__()
        self.freq_size = freq_size
        self.mlp = nn.Sequential(
            nn.Linear(freq_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def _sinusoidal_embedding(self, t: torch.Tensor) -> torch.Tensor:
        """将标量时间 `t` 编码为正余弦特征。"""
        half = self.freq_size // 2
        device = t.device
        dtype = t.dtype
        freq = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=device, dtype=dtype) / max(half - 1, 1)
        )
        args = t[:, None] * freq[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.freq_size % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """将一维时间输入映射到网络隐藏空间。"""
        if t.dim() != 1:
            raise ValueError(f"time must be [B], got {tuple(t.shape)}")
        return self.mlp(self._sinusoidal_embedding(t))


class FeedForward(nn.Module):
    """RDT 对齐版 SiLU-gated FFN。"""

    def __init__(
        self,
        *,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: float | None,
    ) -> None:
        """初始化 SiLU-gated 前馈层。

        参数:
            dim: 输入/输出维度。
            hidden_dim: 中间维度基值。
            multiple_of: 中间维度对齐倍数。
            ffn_dim_multiplier: 可选维度缩放系数。
        """
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """执行 gated FFN 前向计算。"""
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class DiTBlock(nn.Module):
    """RDT 对齐版 DiT block：MSA + Cross-Attention + SiLU-gated FFN。"""

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
        """初始化单层 DiT block（自注意力 + 交叉注意力 + FFN）。

        参数:
            hidden_size: 隐藏维度。
            num_heads: 注意力头数。
            mlp_ratio: FFN 扩展比例。
            norm_eps: 归一化稳定项。
            ffn_multiple_of: FFN 维度对齐倍数。
            ffn_dim_multiplier: 可选 FFN 缩放系数。
        """
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
        """执行单层 DiT 前向。

        参数:
            x: 动作 token 隐状态 `[B,H,C]`。
            ada_cond: AdaLN 条件向量 `[B,2C]`。
            ck: 可选 cross-attention key 条件。
            cv: 可选 cross-attention value 条件。
            attn_mask: 可选 key padding mask（True 表示忽略）。

        返回:
            更新后的动作隐状态。
        """
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
        h = x + gate_attn.unsqueeze(1) * attn_out

        if ck is not None and cv is not None:
            q = _modulate(self.cross_norm(h), shift_cross, scale_cross)
            k = self.cond_norm(ck)
            v = self.cond_norm(cv)
            cross_out, _ = self.cross_attn(q, k, v, key_padding_mask=attn_mask, need_weights=False)
            h = h + gate_cross.unsqueeze(1) * cross_out

        mlp_out = self.ffn(_modulate(self.ffn_norm(h), shift_mlp, scale_mlp))
        return h + gate_mlp.unsqueeze(1) * mlp_out


class ActionExpertBackbone(nn.Module):
    """动作扩散速度场主干网络。"""

    def __init__(self, config: ActionExpertConfig) -> None:
        """构建连续动作速度场网络。"""

        super().__init__()
        self.config = config
        self.action_in = nn.Linear(config.action_dim, config.hidden_size)
        self.state_in = nn.Linear(config.state_dim, config.hidden_size)
        self.state_global_in = nn.Linear(config.state_dim, config.hidden_size)
        self.time_embed = TimestepEmbedder(config.hidden_size, freq_size=config.time_embed_dim)
        self.k_proj = nn.LazyLinear(config.hidden_size)
        self.v_proj = nn.LazyLinear(config.hidden_size)
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
        """将状态张量规范到 `[B,H,Ds]` 形状，便于与动作序列对齐。"""
        if state.dim() == 2:
            state = state.unsqueeze(1)
        if state.shape[1] == 1:
            state = state.expand(batch_size, horizon, -1)
        if state.shape[1] != horizon:
            raise ValueError(f"state horizon mismatch: got {state.shape[1]}, expected {horizon}")
        return state

    @staticmethod
    def _kv_to_tokens(x: torch.Tensor) -> torch.Tensor:
        """将不同 rank 的 KV 张量统一展开为 token 序列格式 `[B,S,C]`。"""
        if x.dim() == 4:
            # 典型 KV 形状: [B, heads, seq, head_dim]
            if x.shape[1] < x.shape[2]:
                x = x.permute(0, 2, 1, 3)
            # 若已是 [B, seq, heads, head_dim]，按原状处理
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
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """将层级 KV 条件投影到动作主干 hidden 空间。

        参数:
            kv_cache: 来自 VLM 的层级 KV 列表。
            batch_size: 当前批大小。
            device: 目标设备。
            dtype: 目标数据类型。

        返回:
            与输入层数对齐的 `(k_proj, v_proj)` 列表。
        """
        if not kv_cache:
            return []
        layer_conds: list[tuple[torch.Tensor, torch.Tensor]] = []
        for k, v in kv_cache:
            if k.shape[0] != batch_size or v.shape[0] != batch_size:
                raise ValueError(
                    f"kv cache batch mismatch: k={tuple(k.shape)}, v={tuple(v.shape)}, expected B={batch_size}"
                )
            k_tokens = self._kv_to_tokens(k.to(device=device, dtype=dtype))
            v_tokens = self._kv_to_tokens(v.to(device=device, dtype=dtype))
            layer_conds.append((self.k_proj(k_tokens), self.v_proj(v_tokens)))
        return layer_conds

    def forward(
        self,
        *,
        noisy_action: torch.Tensor,
        state: torch.Tensor,
        time: torch.Tensor,
        kv_cache: KVCache | None = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        预测扩散速度场 u_t。

        这里的 kv_cache 与 qwen3_stream.kv_export.export_selected_kv_cache 输出协议一致，
        用于在动作头中注入视觉语言上下文偏置。
        """

        batch_size, horizon, _ = noisy_action.shape
        state = self._to_horizon_state(state, batch_size, horizon)
        if time.dim() == 1:
            time = time[:, None]
        if time.shape[1] != 1 and time.shape[1] != horizon:
            raise ValueError(f"time must be [B], [B,1], or [B,H], got {tuple(time.shape)}")
        if time.shape[1] == 1:
            time = time.expand(batch_size, horizon)
        if attention_mask is not None and attention_mask.dim() != 2:
            raise ValueError(f"attention_mask must be [B, S], got {tuple(attention_mask.shape)}")
        if attention_mask is not None and attention_mask.shape[0] != batch_size:
            raise ValueError(
                f"attention_mask batch mismatch: got {attention_mask.shape[0]}, expected {batch_size}"
            )

        x = self.action_in(noisy_action) + self.state_in(state)
        time_cond = self.time_embed(time[:, 0])
        state_cond = self.state_global_in(state.mean(dim=1))
        ada_cond = torch.cat([time_cond, state_cond], dim=-1)

        kv_layer_conds = self._encode_layerwise_kv(
            kv_cache,
            batch_size=batch_size,
            device=noisy_action.device,
            dtype=noisy_action.dtype,
        )

        for i, block in enumerate(self.blocks):
            if kv_layer_conds:
                ck, cv = kv_layer_conds[i % len(kv_layer_conds)]
                if attention_mask is not None:
                    key_padding_mask = self._align_key_padding_mask(
                        attention_mask=attention_mask,
                        kv_len=ck.shape[1],
                        device=ck.device,
                    )
                else:
                    key_padding_mask = None
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
    def _align_key_padding_mask(
        *,
        attention_mask: torch.Tensor,
        kv_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """将上游 1=有效 的 attention_mask 对齐为 MHA 的 key_padding_mask（True=忽略）。"""

        valid_mask = attention_mask.to(device=device)
        if valid_mask.dtype != torch.bool:
            valid_mask = valid_mask > 0

        seq_len = valid_mask.shape[1]
        if seq_len > kv_len:
            valid_mask = valid_mask[:, -kv_len:]
        elif seq_len < kv_len:
            pad = torch.ones((valid_mask.shape[0], kv_len - seq_len), device=device, dtype=torch.bool)
            valid_mask = torch.cat([pad, valid_mask], dim=1)

        return ~valid_mask
