import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from oat.model.common.module_attr_mixin import ModuleAttrMixin


class AutoregressiveModel(ModuleAttrMixin):
    def __init__(self,
        vocab_size: int,
        max_seq_len: int,
        max_cond_len: int,
        cond_dim: int = 0,
        n_layer: int = 12,
        n_head: int = 12,
        n_emb: int = 768,
        p_drop_emb: float = 0.1,
        p_drop_attn: float = 0.1,
    ):
        super().__init__()

        # input embedding stem
        self.tok_emb = nn.Embedding(vocab_size, n_emb)
        self.tok_pos_emb = nn.Parameter(torch.zeros(1, max_seq_len, n_emb))
        self.cond_emb = nn.Linear(cond_dim, n_emb)
        self.cond_pos_emb = nn.Parameter(torch.zeros(1, max_cond_len, n_emb))
        self.drop = nn.Dropout(p_drop_emb)

        self.encoder = nn.Sequential(
            nn.Linear(n_emb, 4 * n_emb),
            nn.Mish(),
            nn.Linear(4 * n_emb, n_emb)
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer=nn.TransformerDecoderLayer(
                d_model=n_emb,
                nhead=n_head,
                dim_feedforward=4*n_emb,
                dropout=p_drop_attn,
                activation='gelu',
                batch_first=True,
                norm_first=True # important for stability
            ),
            num_layers=n_layer
        )
        # decoder head
        self.ln_f = nn.LayerNorm(n_emb)
        self.head = nn.Linear(n_emb, vocab_size, bias=False)

        # tie weights
        self.head.weight = self.tok_emb.weight

        # init
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

    def forward(self, tokens: torch.LongTensor, cond: torch.Tensor):
        """
        tokens: (B,T_tok)
        cond: (B,T_cond,cond_dim)
        output: (B,T,vocab_size)
        """
        T_tok = tokens.shape[1]
        T_cond = cond.shape[1]
        
        # token processing
        tok_emb = self.tok_emb(tokens)
        tok_emb = self.drop(tok_emb + self.tok_pos_emb[:, :T_tok, :])

        # cond processing
        cond_emb = self.cond_emb(cond)
        cond_emb = self.drop(cond_emb + self.cond_pos_emb[:, :T_cond, :])
        cond_emb = self.encoder(cond_emb)

        # causal mask
        tgt_mask = (torch.triu(torch.ones(T_tok, T_tok)) == 1).transpose(0, 1)
        tgt_mask = tgt_mask.float(
            ).masked_fill(tgt_mask == 0, float('-inf')
            ).masked_fill(tgt_mask == 1, float(0.0))

        # transformer decoding
        out = self.decoder(
            tgt=tok_emb,
            memory=cond_emb,
            tgt_mask=tgt_mask,
            memory_mask=None,
        )
        out = self.ln_f(out)
        out = self.head(out)
        return out

    def generate(self, 
        prefix: torch.LongTensor, 
        cond: torch.Tensor, 
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        eos_id: Optional[int] = None,
    ) -> torch.LongTensor:
        """
        Generate tokens autoregressively.
        prefix: (B, T_pre)
        cond: (B, T_cond, cond_dim)
        max_new_tokens: int, max number of new tokens to generate
        temperature: float, sampling temperature
        top_k: Optional[int], if specified, use top-k sampling
        eos_id: Optional[int], if specified, stop generation when eos_id is generated
            all subsequent tokens will be set to eos_id
        output: (B, T_pre + max_new_tokens)
        """
        out_tokens = prefix
        finished = torch.zeros(prefix.size(0), dtype=torch.bool, 
            device=prefix.device) if eos_id is not None else None
        for _ in range(max_new_tokens):
            logits = self.forward(out_tokens, cond)
            logits = logits[:, -1, :]
            if temperature > 0:
                logits = logits / temperature
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')
                probs = F.softmax(logits, dim=-1)  # (B,vocab_size)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)

            # if an EOS token is generated, mark the sequence as finished
            # and replace all subsequent tokens with EOS
            if eos_id is not None:
                if finished.any():
                    next_token = torch.where(
                        finished.view(-1, 1),
                        torch.full_like(next_token, eos_id),
                        next_token
                    )
                finished = finished | (next_token.squeeze(-1) == eos_id)

            out_tokens = torch.cat((out_tokens, next_token), dim=1)

            # If all sequences are finished, stop early
            if eos_id is not None and finished.all():
                break
            
        return out_tokens
