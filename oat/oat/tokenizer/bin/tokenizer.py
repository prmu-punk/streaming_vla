import torch

from oat.tokenizer.base_tokenizer import BaseTokenizer
from oat.model.common.normalizer import LinearNormalizer


class BinTok(BaseTokenizer):
    """
    OpenVLA style binning tokenizer. Adapted from: 
    https://github.com/openvla/openvla/blob/main/prismatic/vla/action_tokenizer.py
    """

    def __init__(self, 
        num_bins: int = 256,
        min_val: float = -1.0,
        max_val: float = 1.0,
    ):
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val
        self.num_bins = num_bins
        self.normalizer = LinearNormalizer()

        bins = torch.linspace(min_val, max_val, num_bins)
        bin_centers = (bins[:-1] + bins[1:]) / 2.0
        self.register_buffer("bins", bins)
        self.register_buffer("bin_centers", bin_centers)
    
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def tokenize(self, samples: torch.Tensor):
        # samples: (B, T, D), floats
        nsamples = self.normalizer["action"](samples)
        nsamples = torch.clamp(nsamples, self.min_val, self.max_val)
        token_ids = torch.bucketize(nsamples, self.bins, right=False)
        token_ids = torch.clamp(token_ids - 1, 0, self.num_bins - 1)
        return token_ids.long()

    def detokenize(self, token_ids: torch.LongTensor):
        # tokens: (B, T, D), integers
        token_ids = torch.clamp(token_ids, 0, self.bin_centers.shape[0] - 1)
        nsamples = self.bin_centers[token_ids]
        samples = self.normalizer["action"].unnormalize(nsamples)
        return samples
