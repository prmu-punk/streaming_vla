
import torch
import torch.nn as nn
import einops
from typing import List, Optional

from oat.tokenizer.oat.model.pos_emb import (
    PositionalEmbedding, PositionalEmbeddingAdder)
from oat.tokenizer.oat.model.head import LinearHead
from oat.tokenizer.oat.model.linear import LinearLayer
from oat.tokenizer.oat.model.token_dropout import MaskedNestedDropout


class SinglePassDecoder(nn.Module):
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
        use_causal_decoder: bool,
        # latent args
        latent_dim: int,
        latent_horizon: int,
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
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=emb_dim,
                nhead=emb_dim // head_dim,
                dim_feedforward=4 * emb_dim,
                dropout=pdropout,
                activation='gelu',
                batch_first=True,
                norm_first=True,
            ),
            num_layers=depth,
        )
        self.latent_proj = LinearLayer(latent_dim, emb_dim)
        self.head = LinearHead(emb_dim, sample_dim)
        
        # attributes
        self.sample_dim = sample_dim
        self.sample_horizon = sample_horizon
        self.latent_horizon = latent_horizon
        self.emb_dim = emb_dim
        self.use_causal_decoder = use_causal_decoder

    def forward(self, 
        latents: torch.Tensor,
        eval_keep_k: Optional[List[int]] = None,
    ) -> torch.Tensor:
        # latents: (B, T', latent_dim)
        # return samples: (B, T, sample_dim) 
        x = self.sample_pos_emb(shape=[self.sample_horizon,]).expand(latents.shape[0], -1, -1)
        x = einops.rearrange(x, "B D T -> B T D")

        latents = self.latent_proj(latents)
        latents = self.latent_pos_emb(latents)
        latents = self.nested_dropout(
            latents, 
            eval_keep_k=eval_keep_k
        )

        if self.use_causal_decoder:
            mask = nn.Transformer.generate_square_subsequent_mask(
                x.size(1), device=x.device
            )
            x = self.decoder(x, latents, tgt_mask=mask, tgt_is_causal=True)
        else:
            x = self.decoder(x, latents)

        samples = self.head(x)  # (B, T, sample_dim)
        return samples
