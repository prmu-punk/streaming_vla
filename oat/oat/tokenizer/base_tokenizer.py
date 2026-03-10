import torch
import dill
import hydra
from oat.model.common.module_attr_mixin import ModuleAttrMixin
from typing import Optional


class BaseTokenizer(ModuleAttrMixin):

    @classmethod
    def from_checkpoint(cls, 
        checkpoint: str, 
        output_dir: Optional[str] = None,
        return_configuration: bool = False,
    ):
        payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
        cfg = payload['cfg']
        cls = hydra.utils.get_class(cfg._target_)
        workspace = cls(cfg, output_dir=output_dir, lazy_instantiation=False)
        workspace.load_payload(payload, exclude_keys=None, include_keys=None)
        tokenizer = workspace.model
        if getattr(cfg.training, "use_ema", False):
            tokenizer = workspace.ema_model
            
        if return_configuration:
            return tokenizer, cfg
        else:
            return tokenizer
    
    def get_optimizer(self, *args, **kwargs) -> torch.optim.Optimizer:
        raise NotImplementedError
    
    def set_normalizer(self, *args, **kwargs):
        raise NotImplementedError
    
    def encode(self, *args, **kwargs):
        raise NotImplementedError
    
    def decode(self, *args, **kwargs):
        raise NotImplementedError
    
    def autoencode(self, *args, **kwargs):
        raise NotImplementedError
    
    def tokenize(self, *args, **kwargs):
        raise NotImplementedError
    
    def detokenize(self, *args, **kwargs):
        raise NotImplementedError
    