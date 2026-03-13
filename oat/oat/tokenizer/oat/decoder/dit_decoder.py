import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from typing import List, Optional, Tuple, Union

from oat.tokenizer.oat.model.pos_emb import (
    PositionalEmbedding, PositionalEmbeddingAdder)
from oat.tokenizer.oat.model.head import LinearHead
from oat.tokenizer.oat.model.linear import LinearLayer
from oat.tokenizer.oat.model.token_dropout import MaskedNestedDropout


def get_safe_dtype(target_dtype, device_type):
    if device_type == "cpu":
        if target_dtype == torch.bfloat16:
            return torch.float32
        if target_dtype == torch.float64:
            return torch.float64
    return target_dtype


def create_sinusoidal_pos_embedding(
    time: torch.Tensor, dimension: int, min_period: float, max_period: float, device="cpu"
) -> torch.Tensor:
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    # Compute the outer product
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


def sample_beta(alpha, beta, bsize, device):
    alpha_t = torch.as_tensor(alpha, dtype=torch.float32, device=device)
    beta_t = torch.as_tensor(beta, dtype=torch.float32, device=device)
    dist = torch.distributions.Beta(alpha_t, beta_t)
    return dist.sample((bsize,))


class DiTBlock(nn.Module):
    def __init__(self, emb_dim: int, num_heads: int, mlp_ratio: float, pdropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(emb_dim, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(emb_dim, elementwise_affine=False)
        self.self_attn = nn.MultiheadAttention(
            emb_dim, num_heads, dropout=pdropout, batch_first=True
        )
        hidden_dim = int(emb_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            LinearLayer(emb_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(pdropout),
            LinearLayer(hidden_dim, emb_dim),
            nn.Dropout(pdropout),
        )
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, 6 * emb_dim),
        )
        # DiT Zero-Initialization
        nn.init.constant_(self.modulation[-1].weight, 0)
        nn.init.constant_(self.modulation[-1].bias, 0)

    def _modulate(self, x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        return x * (1 + scale[:, None, :]) + shift[:, None, :]

    def forward(
        self,
        x: torch.Tensor,
        t_emb: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.modulation(t_emb).chunk(6, dim=1)
        x_norm = self._modulate(self.norm1(x), shift_msa, scale_msa)
        attn_out = self.self_attn(
            x_norm, x_norm, x_norm,
            attn_mask=attn_mask,
            need_weights=False,
            is_causal=is_causal if attn_mask is None else False
        )[0]
        x = x + gate_msa[:, None, :] * attn_out
        x_norm = self._modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp[:, None, :] * self.mlp(x_norm)
        return x


class CrossAttnDiTBlock(nn.Module):
    def __init__(self, emb_dim: int, num_heads: int, mlp_ratio: float, pdropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(emb_dim, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(emb_dim, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(emb_dim, elementwise_affine=False)
        self.self_attn = nn.MultiheadAttention(
            emb_dim, num_heads, dropout=pdropout, batch_first=True
        )
        self.cross_attn = nn.MultiheadAttention(
            emb_dim, num_heads, dropout=pdropout, batch_first=True
        )
        hidden_dim = int(emb_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            LinearLayer(emb_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(pdropout),
            LinearLayer(hidden_dim, emb_dim),
            nn.Dropout(pdropout),
        )
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, 9 * emb_dim),
        )
        nn.init.constant_(self.modulation[-1].weight, 0)
        nn.init.constant_(self.modulation[-1].bias, 0)

    def _modulate(self, x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        return x * (1 + scale[:, None, :]) + shift[:, None, :]

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        t_emb: torch.Tensor,
        self_attn_mask: Optional[torch.Tensor] = None,
        cross_attn_mask: Optional[torch.Tensor] = None,
        self_is_causal: bool = False,
    ) -> torch.Tensor:
        mods = self.modulation(t_emb).chunk(9, dim=1)
        shift_sa, scale_sa, gate_sa, shift_ca, scale_ca, gate_ca, shift_mlp, scale_mlp, gate_mlp = mods
        x_norm = self._modulate(self.norm1(x), shift_sa, scale_sa)
        attn_out = self.self_attn(
            x_norm, x_norm, x_norm,
            attn_mask=self_attn_mask,
            need_weights=False,
            is_causal=self_is_causal if self_attn_mask is None else False
        )[0]
        x = x + gate_sa[:, None, :] * attn_out
        x_norm = self._modulate(self.norm2(x), shift_ca, scale_ca)
        attn_out = self.cross_attn(
            x_norm, context, context,
            attn_mask=cross_attn_mask,
            need_weights=False
        )[0]
        x = x + gate_ca[:, None, :] * attn_out
        x_norm = self._modulate(self.norm3(x), shift_mlp, scale_mlp)
        x = x + gate_mlp[:, None, :] * self.mlp(x_norm)
        return x


class DiTDecoder(nn.Module):
    def __init__(self,
        # sample attrs
        sample_dim: int,
        sample_horizon: int,
        # decoder args
        emb_dim: int,
        head_dim: int, 
        depth: int,
        pdropout: float,
        token_dropout_mode: str,
        mask_type: str,
        # latent args
        latent_dim: int,
        latent_horizon: int,
        # dit args
        diffusion_steps: int = 10,
        time_emb_min_period: float = 4.0e-3,
        time_emb_max_period: float = 4.0,
        time_emb_scale: float = 1.0,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        
        self.sample_pos_emb = PositionalEmbedding(
            emb_dim,
            max_sizes=[sample_horizon,]
        )
        self.latent_pos_emb = PositionalEmbeddingAdder(
            emb_dim,
            max_sizes=[latent_horizon,]
        )
        self.nested_dropout = MaskedNestedDropout(
            emb_dim,
            size_sampling_mode=token_dropout_mode
        )
        num_heads = emb_dim // head_dim
        self.blocks = nn.ModuleList([
            DiTBlock(emb_dim, num_heads, mlp_ratio, pdropout)
            for _ in range(depth)
        ])
        self.latent_proj = LinearLayer(latent_dim, emb_dim)
        
        # Final AdaLN
        self.final_norm = nn.LayerNorm(emb_dim, elementwise_affine=False)
        self.final_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, 2 * emb_dim) # shift and scale
        )
        # Zero-initialize final modulation
        nn.init.constant_(self.final_modulation[-1].weight, 0)
        nn.init.constant_(self.final_modulation[-1].bias, 0)

        self.head = LinearHead(emb_dim, sample_dim)
        
        # DiT Components
        self.diffusion_steps = diffusion_steps
        
        # Project noisy sample to embedding dimension
        self.noisy_sample_proj = LinearLayer(sample_dim, emb_dim)
        
        # Time Embedding MLP
        self.time_mlp = nn.Sequential(
            LinearLayer(emb_dim, emb_dim),
            nn.SiLU(),
            LinearLayer(emb_dim, emb_dim)
        )
        self.time_emb_min_period = time_emb_min_period
        self.time_emb_max_period = time_emb_max_period
        self.time_emb_scale = time_emb_scale
        
        # attributes
        self.sample_dim = sample_dim
        self.sample_horizon = sample_horizon
        self.latent_horizon = latent_horizon
        self.emb_dim = emb_dim
        self.mask_type = mask_type

    def create_prefix_mask(
        self,
        T_lat: int,
        T_samp: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        T_total = T_lat + T_samp
        mask = torch.zeros((T_total, T_total), device=device, dtype=dtype)
        mask[:T_lat, T_lat:] = float("-inf")
        if self.mask_type in ["causal", "oat"]:
            latent_causal = nn.Transformer.generate_square_subsequent_mask(T_lat, device=device)
            if latent_causal.dtype != dtype:
                latent_causal = latent_causal.to(dtype)
            mask[:T_lat, :T_lat] = latent_causal
        if self.mask_type == "causal":
            sample_causal = nn.Transformer.generate_square_subsequent_mask(T_samp, device=device)
            if sample_causal.dtype != dtype:
                sample_causal = sample_causal.to(dtype)
            mask[T_lat:, T_lat:] = sample_causal
        return mask

    def forward(self, 
        latents: torch.Tensor,
        eval_keep_k: Optional[List[int]] = None,
        target: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # latents: (B, T', latent_dim)
        # target: (B, T, sample_dim) - Ground Truth for training
        
        if target is not None:
            return self._train_forward(latents, target, eval_keep_k)
        else:
            return self._inference_sample(latents, eval_keep_k)

    def _forward_model(self, latents, noisy_sample, t, eval_keep_k, attn_mask: Optional[torch.Tensor] = None):
        # 1. Process Latents (Condition)
        latents = self.latent_proj(latents)
        latents = self.latent_pos_emb(latents)
        latents = self.nested_dropout(
            latents, 
            eval_keep_k=eval_keep_k
        )

        # 2. Process Noisy Input
        x = self.noisy_sample_proj(noisy_sample) # (B, T, emb_dim)
        
        # Add Spatial Positional Embedding
        pos_emb = self.sample_pos_emb(shape=[self.sample_horizon,]).expand(x.shape[0], -1, -1)
        pos_emb = einops.rearrange(pos_emb, "B D T -> B T D")
        x = x + pos_emb
        
        # Concatenate Latents and Noisy Sample
        # latents: (B, T_lat, D), x: (B, T_samp, D)
        # combined: (B, T_lat + T_samp, D)
        x_combined = torch.cat([latents, x], dim=1)
        
        # Add Time Embedding
        t_input = t * self.time_emb_scale
        t_emb = create_sinusoidal_pos_embedding(
            t_input,
            self.emb_dim,
            self.time_emb_min_period,
            self.time_emb_max_period,
            device=x.device,
        )
        t_emb = t_emb.to(dtype=x.dtype)
        t_emb = self.time_mlp(t_emb) # (B, emb_dim)
        
        if attn_mask is None:
            T_lat = latents.shape[1]
            T_samp = x.shape[1]
            if not hasattr(self, "_cached_train_mask") or \
               self._cached_train_mask.device != x.device or \
               self._cached_train_mask.dtype != x.dtype or \
               self._cached_train_mask.shape[0] != T_lat + T_samp or \
               getattr(self, "_cached_T_lat", -1) != T_lat or \
               getattr(self, "_cached_T_samp", -1) != T_samp:
                self._cached_train_mask = self.create_prefix_mask(T_lat, T_samp, x.device, x.dtype)
                self._cached_T_lat = T_lat
                self._cached_T_samp = T_samp
            attn_mask = self._cached_train_mask

        for block in self.blocks:
            x_combined = block(x_combined, t_emb, attn_mask=attn_mask)

        # Final AdaLN
        x_out = x_combined[:, -self.sample_horizon:, :]
        shift_final, scale_final = self.final_modulation(t_emb).chunk(2, dim=1)
        x_out = self.final_norm(x_out) * (1 + scale_final[:, None, :]) + shift_final[:, None, :]

        # 4. Predict Velocity
        v_pred = self.head(x_out)  # (B, T, sample_dim)
        return v_pred

    def _train_forward(self, latents, target, eval_keep_k):
        B = target.shape[0]
        device = target.device
        dtype = target.dtype
        
        # Sample Time & Noise
        t = sample_beta(1.5, 1.0, B, device) # (B,)
        t = t * 0.999 + 0.001 # Avoid 0 and 1
        t = t.to(dtype=dtype)
        
        noise = torch.randn_like(target)
        
        # Flow Matching Interpolation: x_t = t * noise + (1-t) * x0
        t_expanded = t.view(B, 1, 1)
        noisy_sample = t_expanded * noise + (1 - t_expanded) * target
        
        # Predict Velocity
        pred_v = self._forward_model(latents, noisy_sample, t, eval_keep_k)
        
        # Calculate Target Velocity for Flow Matching
        # v_t = noise - x0
        target_v = noise - target
        
        return pred_v, target_v

    def _inference_sample(self, latents, eval_keep_k):
        B = latents.shape[0]
        device = latents.device
        dtype = latents.dtype
        T = self.sample_horizon
        D = self.sample_dim
        
        # Start from pure noise (t=1)
        x_t = torch.randn(B, T, D, device=device, dtype=dtype)
        
        # Euler Integration (t=1 -> t=0)
        dt = -1.0 / self.diffusion_steps
        curr_t = 1.0
        
        # Pre-compute mask to avoid re-computation in the loop
        T_lat = latents.shape[1]
        T_samp = self.sample_horizon
        attn_mask = self.create_prefix_mask(T_lat, T_samp, device, dtype)
        
        for i in range(self.diffusion_steps):
            t_batch = torch.full((B,), curr_t, device=device, dtype=x_t.dtype)
            
            # Predict velocity
            v_pred = self._forward_model(latents, x_t, t_batch, eval_keep_k, attn_mask=attn_mask)
            
            # Update x
            x_t = x_t + dt * v_pred
            curr_t += dt
            
        return x_t


class CrossAttnDiTDecoder(nn.Module):
    def __init__(self,
        sample_dim: int,
        sample_horizon: int,
        emb_dim: int,
        head_dim: int,
        depth: int,
        pdropout: float,
        token_dropout_mode: str,
        mask_type: str,
        latent_dim: int,
        latent_horizon: int,
        diffusion_steps: int = 10,
        time_emb_min_period: float = 4.0e-3,
        time_emb_max_period: float = 4.0,
        time_emb_scale: float = 1.0,
        mlp_ratio: float = 4.0,
        state_dim: int | None = None,
    ):
        super().__init__()
        del state_dim

        self.sample_pos_emb = PositionalEmbedding(
            emb_dim,
            max_sizes=[sample_horizon,]
        )
        self.latent_pos_emb = PositionalEmbeddingAdder(
            emb_dim,
            max_sizes=[latent_horizon,]
        )
        self.nested_dropout = MaskedNestedDropout(
            emb_dim,
            size_sampling_mode=token_dropout_mode
        )
        num_heads = emb_dim // head_dim
        self.blocks = nn.ModuleList([
            CrossAttnDiTBlock(emb_dim, num_heads, mlp_ratio, pdropout)
            for _ in range(depth)
        ])
        self.latent_proj = LinearLayer(latent_dim, emb_dim)
        self.final_norm = nn.LayerNorm(emb_dim, elementwise_affine=False)
        self.final_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, 2 * emb_dim)
        )
        nn.init.constant_(self.final_modulation[-1].weight, 0)
        nn.init.constant_(self.final_modulation[-1].bias, 0)
        self.head = LinearHead(emb_dim, sample_dim)
        self.diffusion_steps = diffusion_steps
        self.noisy_sample_proj = LinearLayer(sample_dim, emb_dim)
        self.time_mlp = nn.Sequential(
            LinearLayer(emb_dim, emb_dim),
            nn.SiLU(),
            LinearLayer(emb_dim, emb_dim)
        )
        self.time_emb_min_period = time_emb_min_period
        self.time_emb_max_period = time_emb_max_period
        self.time_emb_scale = time_emb_scale
        self.sample_dim = sample_dim
        self.sample_horizon = sample_horizon
        self.latent_horizon = latent_horizon
        self.emb_dim = emb_dim
        self.mask_type = mask_type

    def _get_sample_mask(
        self,
        T_samp: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if not hasattr(self, "_cached_sample_mask") or \
           self._cached_sample_mask.device != device or \
           self._cached_sample_mask.dtype != dtype or \
           self._cached_sample_mask.shape[0] != T_samp or \
           getattr(self, "_cached_T_samp", -1) != T_samp:
            mask = nn.Transformer.generate_square_subsequent_mask(T_samp, device=device)
            if mask.dtype != dtype:
                mask = mask.to(dtype)
            self._cached_sample_mask = mask
            self._cached_T_samp = T_samp
        return self._cached_sample_mask

    def forward(
        self,
        latents: torch.Tensor,
        eval_keep_k: Optional[List[int]] = None,
        target: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if target is not None:
            return self._train_forward(latents, target, eval_keep_k)
        return self._inference_sample(latents, eval_keep_k)

    def _forward_model(self, latents, noisy_sample, t, eval_keep_k, self_attn_mask: Optional[torch.Tensor] = None):
        latents = self.latent_proj(latents)
        latents = self.latent_pos_emb(latents)
        latents = self.nested_dropout(
            latents,
            eval_keep_k=eval_keep_k
        )

        x = self.noisy_sample_proj(noisy_sample)
        pos_emb = self.sample_pos_emb(shape=[self.sample_horizon,]).expand(x.shape[0], -1, -1)
        pos_emb = einops.rearrange(pos_emb, "B D T -> B T D")
        x = x + pos_emb

        t_input = t * self.time_emb_scale
        t_emb = create_sinusoidal_pos_embedding(
            t_input,
            self.emb_dim,
            self.time_emb_min_period,
            self.time_emb_max_period,
            device=x.device,
        )
        t_emb = t_emb.to(dtype=x.dtype)
        t_emb = self.time_mlp(t_emb)

        sample_mask = None
        if self.mask_type == "causal":
            if self_attn_mask is None:
                sample_mask = self._get_sample_mask(x.shape[1], x.device, x.dtype)
            else:
                sample_mask = self_attn_mask

        for block in self.blocks:
            x = block(
                x,
                latents,
                t_emb,
                self_attn_mask=sample_mask,
                cross_attn_mask=None,
                self_is_causal=self.mask_type == "causal"
            )

        shift_final, scale_final = self.final_modulation(t_emb).chunk(2, dim=1)
        x = self.final_norm(x) * (1 + scale_final[:, None, :]) + shift_final[:, None, :]
        v_pred = self.head(x)
        return v_pred

    def _train_forward(self, latents, target, eval_keep_k):
        B = target.shape[0]
        device = target.device
        dtype = target.dtype
        t = sample_beta(1.5, 1.0, B, device)
        t = t * 0.999 + 0.001
        t = t.to(dtype=dtype)
        noise = torch.randn_like(target)
        t_expanded = t.view(B, 1, 1)
        noisy_sample = t_expanded * noise + (1 - t_expanded) * target
        pred_v = self._forward_model(latents, noisy_sample, t, eval_keep_k)
        target_v = noise - target
        return pred_v, target_v

    def _inference_sample(self, latents, eval_keep_k):
        B = latents.shape[0]
        device = latents.device
        dtype = latents.dtype
        T = self.sample_horizon
        D = self.sample_dim
        x_t = torch.randn(B, T, D, device=device, dtype=dtype)
        dt = -1.0 / self.diffusion_steps
        curr_t = 1.0
        sample_mask = None
        if self.mask_type == "causal":
            sample_mask = self._get_sample_mask(T, device, dtype)
        for _ in range(self.diffusion_steps):
            t_batch = torch.full((B,), curr_t, device=device, dtype=x_t.dtype)
            v_pred = self._forward_model(latents, x_t, t_batch, eval_keep_k, self_attn_mask=sample_mask)
            x_t = x_t + dt * v_pred
            curr_t += dt
        return x_t
