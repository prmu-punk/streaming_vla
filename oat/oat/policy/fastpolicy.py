import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List

from oat.policy.base_policy import BasePolicy
from oat.tokenizer.fast.tokenizer_wrapper import FASTTok
from oat.perception.base_obs_encoder import BaseObservationEncoder
# from oat.model.autoregressive.transformer import AutoregressiveModel
from oat.model.autoregressive.transformer_cache import AutoregressiveModel


class FASTPolicy(BasePolicy):
    def __init__(
        self,
        shape_meta: Dict,
        obs_encoder: BaseObservationEncoder,
        action_tokenizer: FASTTok,
        horizon: int,
        n_action_steps: int,
        n_obs_steps: int,
        # policy model params
        embed_dim: int = 512,
        n_layers: int = 8,
        n_heads: int = 8,
        dropout: float = 0.1,
        # policy inference params
        max_seq_len: int = 128,
        temperature: float = 1.0,
        topk: int = 10,
    ):
        super().__init__()
        
        modalities = obs_encoder.modalities()
        obs_feature_dim = obs_encoder.output_feature_dim()
        action_shape = shape_meta["action"]["shape"]
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        obs_key_shapes = dict()
        obs_ports = []
        for key, attr in shape_meta['obs'].items():
            shape = attr['shape']
            obs_key_shapes[key] = list(shape)
            type = attr['type']
            if type in modalities:
                obs_ports.append(key)

        # freeze action tokenizer
        for param in action_tokenizer.parameters():
            param.requires_grad_(False)
        action_tokenizer.eval()

        # special token ids
        codebook_size = action_tokenizer.vocab_size
        bos_id = codebook_size
        eos_id = codebook_size + 1
        pad_id = codebook_size + 2

        # create AR model
        model = AutoregressiveModel(
            vocab_size=codebook_size + 3,   # + <BOS>, <EOS>, <PAD>
            max_seq_len=max_seq_len + 1,    # + <BOS>
            max_cond_len=n_obs_steps,
            cond_dim=obs_feature_dim,
            n_layer=n_layers,
            n_head=n_heads,
            n_emb=embed_dim,
            p_drop_emb=dropout,
            p_drop_attn=dropout,
        )

        self.modalities = modalities
        self.obs_key_shapes = obs_key_shapes
        self.obs_ports = obs_ports
        self.obs_encoder = obs_encoder
        self.action_tokenizer = action_tokenizer
        self.model = model
        self.max_seq_len = max_seq_len
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.pad_id = pad_id
        self.horizon = horizon
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.temperature = temperature
        self.topk = topk

        # report
        num_obs_params = sum(p.numel() for p in obs_encoder.parameters())
        num_trainable_obs_params = sum(p.numel() for p in obs_encoder.parameters() if p.requires_grad)
        obs_trainable_ratio = num_trainable_obs_params / num_obs_params if num_obs_params > 0 else 0.0
        num_tok_params = sum(p.numel() for p in action_tokenizer.parameters())
        num_trainable_tok_params = sum(p.numel() for p in action_tokenizer.parameters() if p.requires_grad)
        tok_trainable_ratio = num_trainable_tok_params / num_tok_params if num_tok_params > 0 else 0.0
        num_model_params = sum(p.numel() for p in model.parameters())
        num_trainable_model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        model_trainable_ratio = num_trainable_model_params / num_model_params if num_model_params > 0 else 0.0
        print(
            f"{self.get_policy_name()} initialized with\n"
            f"  obs enc: {num_obs_params/1e6:.1f}M ({obs_trainable_ratio:.5%} trainable)\n"
            f"  act tok: {num_tok_params/1e6:.1f}M ({tok_trainable_ratio:.5%} trainable)\n"
            f"  policy : {num_model_params/1e6:.1f}M ({model_trainable_ratio:.5%} trainable)\n"
        )

    def get_observation_encoder(self):
        return self.obs_encoder

    def get_observation_modalities(self):
        return self.modalities
    
    def get_observation_ports(self):
        return self.obs_ports
    
    def get_policy_name(self):
        base_name = 'fastpolicy_'
        for modality in self.modalities:
            if modality != 'state':
                base_name += modality + '|'
        return base_name[:-1]

    def create_dummy_observation(self,
        batch_size: int = 1,
        device: Optional[torch.device] = None
    ) -> Dict[str, torch.Tensor]:
        return super().create_dummy_observation(
            batch_size=batch_size,
            horizon=self.n_obs_steps,
            obs_key_shapes=self.obs_key_shapes,
            device=device
        )

    def set_normalizer(self, normalizer):
        self.obs_encoder.set_normalizer(normalizer)
        # self.action_tokenizer.set_normalizer(normalizer)

    def get_optimizer(
        self, 
        policy_lr: float,
        obs_enc_lr: float,
        weight_decay: float,
        betas: Tuple[float, float],
    ) -> torch.optim.Optimizer:
        """Create an AdamW optimizer with weight decay for 2D parameters only."""
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.

        encoder_decay_params = []
        encoder_nodecay_params = []
        for name, param in self.obs_encoder.named_parameters():
            if not param.requires_grad:
                continue
            if param.dim() >= 2:
                encoder_decay_params.append(param)
            else:
                encoder_nodecay_params.append(param)

        policy_decay_params = []
        policy_nodecay_params = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if param.dim() >= 2:
                policy_decay_params.append(param)
            else:
                policy_nodecay_params.append(param)
        
        optim_groups = [
            {'params': policy_decay_params, 'lr': policy_lr, 'weight_decay': weight_decay},
            {'params': policy_nodecay_params, 'lr': policy_lr, 'weight_decay': 0.0},
            {'params': encoder_decay_params, 'lr': obs_enc_lr, 'weight_decay': weight_decay},
            {'params': encoder_nodecay_params, 'lr': obs_enc_lr, 'weight_decay': 0.0},
        ]

        optimizer = torch.optim.AdamW(optim_groups, betas=betas)
        return optimizer

    def predict_action(self, 
        obs_dict: Dict[str, torch.Tensor],
        temperature: Optional[float] = None,
        topk: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        if temperature is None:
            temperature = self.temperature
        if topk is None:
            topk = self.topk

        # encode observation
        features = self.obs_encoder(obs_dict)   # [B, To, d]
        B = features.shape[0]

        # autoregressive generation
        action_tokens = torch.full( # [B, 1] seq: [<BOS>,]
            (B, 1), self.bos_id, 
            dtype=torch.long, device=self.device
        ) 
        action_tokens = self.model.generate(
            action_tokens,
            cond=features,
            max_new_tokens=self.max_seq_len,
            temperature=temperature,
            top_k=topk,
            eos_id=self.eos_id,
        )[:, 1:]    # [B, L'] with <EOS> padding
        B, L = action_tokens.shape

        # trim <EOS> padding
        idx = torch.arange(L, device=self.device).unsqueeze(0).expand(B, L)
        is_eos = action_tokens.eq(self.eos_id)
        first_eos = torch.where(is_eos, idx, L).amin(dim=1)  # [B,]
        has_eos = first_eos.lt(L)
        cut = torch.where(has_eos, first_eos, torch.full_like(first_eos, L)).tolist()  # [B,]
        action_tokens = action_tokens.detach().cpu()
        action_tokens: List[List[int]] = [action_tokens[b, :cut[b]].tolist() for b in range(B)]

        # decode action tokens
        with torch.inference_mode():
            action_pred = self.action_tokenizer.detokenize(
                tokens=action_tokens,
                horizon=self.horizon,
                dim=self.action_dim,
            )   # [B, horizon, Da]

        # receeding horizon
        action = action_pred[:,:self.n_action_steps]

        result = {
            'action': action,
            'action_pred': action_pred
        }
        return result


    def forward(self, batch) -> torch.Tensor:
        B = batch['action'].shape[0]
        device = batch['action'].device

        # tokenize trajectory
        with torch.inference_mode():
            action_tokens: List[List[int]] = self.action_tokenizer.tokenize(batch['action'])
        
        # token seq padding
        L = self.max_seq_len
        ignore_idx = -100
        lens = torch.tensor([len(t) for t in action_tokens], device=device)
        lens = torch.clamp(lens, max=L-1) # -1 for <EOS>
        action_tokens = [
            torch.tensor(seq[:seq_len.item()], device=device, dtype=torch.long)
            for seq, seq_len in zip(action_tokens, lens)
        ]

        # append <EOS>
        eos = torch.tensor([self.eos_id], device=device, dtype=torch.long)
        action_tokens = [
            torch.cat([core, eos], dim=0)
            for core in action_tokens
        ]

        # post pad <PAD> to L
        action_tokens = nn.utils.rnn.pad_sequence(
            action_tokens, 
            batch_first=True, 
            padding_value=self.pad_id
        )   # [B, L']
        if action_tokens.shape[-1] < L:
            action_tokens = F.pad(
                action_tokens,
                (0, L - action_tokens.shape[-1]),
                value=self.pad_id
            )
        else:
            action_tokens = action_tokens[:, :L]
        assert action_tokens.shape == (B, L)

        # prepend <BOS>
        action_tokens = torch.cat([
            torch.full((B, 1), self.bos_id, dtype=torch.long, device=device), 
            action_tokens
        ], dim=1)   # [B, L+1]

        # construct input & target ids
        input_ids = action_tokens[:,:-1]    # [B, L]  <BOS> ... <EOS> <PAD> <PAD> ...
        target_ids = action_tokens[:,1:]    # [B, L]  ... <EOS> <PAD> <PAD> ... <PAD>
        target_ids = target_ids.masked_fill(
            target_ids.eq(self.pad_id), ignore_idx
        )   # [B, L]  ... <EOS> <IGNORE> <IGNORE> ... <IGNORE>
            
        # encode observation
        features = self.obs_encoder(batch['obs'])   # [B, To, d]

        # forward model
        logits = self.model(input_ids, cond=features)

        # compute loss
        vocab_size = logits.size(-1)
        loss = F.cross_entropy(
            logits.reshape(-1, vocab_size),     # (B*T, vocab_size)
            target_ids.reshape(-1),             # (B*T,)
            ignore_index=ignore_idx,
        )
        return loss
