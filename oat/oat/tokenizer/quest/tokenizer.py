# Adapted from https://github.com/pairlab/QueST
# -----------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from vector_quantize_pytorch import VectorQuantize, FSQ
from typing import Literal, List, Tuple

from oat.tokenizer.base_tokenizer import BaseTokenizer
from oat.model.common.normalizer import LinearNormalizer
from oat.tokenizer.quest.model.pos_emb import PositionalEncoding1D, Summer
from oat.tokenizer.quest.model.conv import ResidualTemporalBlock


class QueSTTok(BaseTokenizer):
    """
    Supports vanilla VQ and FSQ
    """

    def __init__(
        self,
        # action chunk related params
        action_dim: int,
        horizon: int,
        # model related params
        encoder_dim: int = 256,
        encoder_nhead: int = 4,
        encoder_nlayer: int = 2,
        use_causal_encoder: bool = True,
        decoder_dim: int = 256,
        decoder_nhead: int = 4,
        decoder_nlayer: int = 4,
        use_causal_decoder: bool = True,
        downsample_factor: int = 4,
        dropout: float = 0.1,
        # vector-quantization related params
        vq_type: Literal['vq', 'fsq'] = 'fsq',
        fsq_level: List[int] = [8, 5, 5, 5],
        vq_codebook_size: int = 1024,
        vq_codebook_dim: int = 512,
    ):
        super().__init__()
        assert (
            int(np.log2(downsample_factor)) == 
            np.log2(downsample_factor)
        ), 'downsample_factor must be a power of 2'

        strides = [2] * int(np.log2(downsample_factor)) + [1]
        kernel_sizes = [5] + [3] * int(np.log2(downsample_factor))
        if len(strides) == 1:
            kernel_sizes = [3, 2]
            strides = [1,1]

        if vq_type == 'vq':
            self.vq = VectorQuantize(dim=encoder_dim, 
                codebook_dim=vq_codebook_dim, 
                codebook_size=vq_codebook_size
            )
            self.codebook_size = vq_codebook_size
        elif vq_type == 'fsq':
            self.vq = FSQ(dim=encoder_dim, levels=fsq_level)
            self.codebook_size = np.prod(fsq_level)
        else:
            raise NotImplementedError('only vq and fsq are supported')
        
        # action projector
        self.action_proj = nn.Linear(action_dim, encoder_dim)
        self.action_head = nn.Linear(decoder_dim, action_dim)

        # conv block
        self.conv_block = ResidualTemporalBlock(
            encoder_dim, encoder_dim, 
            kernel_size=kernel_sizes,
            stride=strides,
            causal=use_causal_encoder,
        )

        # normalizer
        self.normalizer = LinearNormalizer()

        # encoder & decoder
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=encoder_dim, 
                nhead=encoder_nhead,
                dim_feedforward=4 * encoder_dim,
                dropout=dropout,
                activation='gelu',
                batch_first=True,
                norm_first=True,
            ),
            num_layers=encoder_nlayer,
            enable_nested_tensor=False,
        )
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=decoder_dim, 
                nhead=decoder_nhead,
                dim_feedforward=4 * decoder_dim,
                dropout=dropout,
                activation='gelu',
                batch_first=True,
                norm_first=True,
            ),
            num_layers=decoder_nlayer,
        )

        # pos emb
        self.add_pos_emb = Summer(PositionalEncoding1D(encoder_dim))
        self.fix_pos_emb = PositionalEncoding1D(decoder_dim)

        # attr
        self.horizon = horizon
        self.token_seq_len = horizon // downsample_factor
        self.use_causal_encoder = use_causal_encoder
        self.decoder_dim = decoder_dim
        self.use_causal_decoder = use_causal_decoder
        self.vq_type = vq_type

        self.init_weights_sp()

    def init_weights_sp(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.xavier_uniform_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def get_optimizer(
        self, 
        learning_rate: float,
        weight_decay: float,
        betas: Tuple[float, float],
    ) -> torch.optim.Optimizer:
        """Create an AdamW optimizer with weight decay for 2D parameters only."""
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in self.named_parameters() if p.requires_grad and p.dim() >= 2]
        nodecay_params = [p for n, p in self.named_parameters() if p.requires_grad and p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)
        return optimizer

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def forward(self, batch) -> torch.Tensor:
        actions = batch['action']
        quants, _, commit_loss = self.encode(actions)
        recons = self.decode(quants)
        recon_loss = F.mse_loss(recons, actions)
        return recon_loss + commit_loss
    
    def encode(self, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # actions: (B, T, d)
        # normalize
        x = self.normalizer["action"].normalize(actions)
        # encode
        x = self.action_proj(x)
        x = self.conv_block(x)
        x = self.add_pos_emb(x)
        if self.use_causal_encoder:
            mask = nn.Transformer.generate_square_subsequent_mask(
                x.size(1), device=x.device)
            x = self.encoder(x, mask=mask)
        else:
            x = self.encoder(x)
        # quantization
        if self.vq_type == 'vq':
            quants, indices, commit_loss = self.vq(x)
        else:
            quants, indices = self.vq(x)
            commit_loss = torch.tensor(0.0, device=x.device)
        return quants, indices, commit_loss

    def decode(self, quants: torch.Tensor) -> torch.Tensor:
        # quants: (B, T', encoder_dim)
        x = self.fix_pos_emb(
            torch.zeros(
                (quants.shape[0], self.horizon, self.decoder_dim), 
                dtype=quants.dtype, device=quants.device
            )
        )
        if self.use_causal_decoder:
            mask = nn.Transformer.generate_square_subsequent_mask(
                x.size(1), device=x.device)
            x = self.decoder(x, quants, tgt_mask=mask, tgt_is_causal=True)
        else:
            x = self.decoder(x, quants)
        x = self.action_head(x)
        # unnormalize
        x = self.normalizer["action"].unnormalize(x)
        return x    # (B, T, action_dim)

    def autoencode(self, actions: torch.Tensor) -> torch.Tensor:
        quants, _, _ = self.encode(actions)
        recons = self.decode(quants)
        return recons

    def tokenize(self, actions: torch.Tensor) -> torch.Tensor:
        _, indices, _ = self.encode(actions)
        return indices

    def detokenize(self, indices: torch.Tensor) -> torch.Tensor:
        if self.vq_type == 'vq':
            quants = self.vq.get_output_from_indices(indices)
        else:
            quants = self.vq.indices_to_codes(indices)
        recons = self.decode(quants)
        return recons

