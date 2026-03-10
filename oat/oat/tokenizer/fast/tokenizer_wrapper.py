import os
import dill
import hydra
import torch
from transformers import AutoProcessor
from typing import List, Optional

from oat.tokenizer.base_tokenizer import BaseTokenizer
from oat.model.common.normalizer import LinearNormalizer


os.environ["TOKENIZERS_PARALLELISM"] = "false"
class FASTTok(BaseTokenizer):
    def __init__(self, fast_tokenizer_path: str = "physical-intelligence/fast"):
        super().__init__()
        self.fast_tok = AutoProcessor.from_pretrained(
            fast_tokenizer_path, trust_remote_code=True)
        self.vocab_size = self.fast_tok.vocab_size
        self.normalizer = LinearNormalizer()

    @classmethod
    def from_checkpoint(cls, 
        checkpoint: str, 
        output_dir: Optional[str] = None,
        return_configuration: bool = False,
    ):
        tok, cfg = super().from_checkpoint(
            checkpoint, output_dir=output_dir, return_configuration=True
        )
        fast_path = os.path.join(
            os.path.dirname(checkpoint), cfg.checkpoint.fast_save_name
        )
        tok.fast_tok = AutoProcessor.from_pretrained(
            fast_path, trust_remote_code=True)
        tok.vocab_size = tok.fast_tok.vocab_size
        
        if return_configuration:
            return tok, cfg
        else:
            return tok

    def set_normalizer(self, normalizer: LinearNormalizer):
        # normalize to [-1, 1], see doc below
        # https://huggingface.co/physical-intelligence/fast
        self.normalizer.load_state_dict(normalizer.state_dict())
        
    def tokenize(self, samples: torch.Tensor):
        # samples: (B, T, D), floats
        # return List[List[int]] of token ids, can be different lengths
        nsamples = self.normalizer["action"].normalize(samples)
        return self.fast_tok(nsamples.cpu().numpy())
    
    def detokenize(self, 
        tokens: List[List[int]],
        horizon: Optional[int] = None,
        dim: Optional[int] = None
    ):
        # tokens: List[List[int]] of token ids
        # return (B, T, D), floats
        nsamples = torch.from_numpy(self.fast_tok.decode(
            tokens, time_horizon=horizon, action_dim=dim
        )).to(dtype=torch.float32, device=self.device)
        samples = self.normalizer["action"].unnormalize(nsamples)
        return samples
