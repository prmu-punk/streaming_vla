from omegaconf import OmegaConf

def register_new_resolvers():
    # OmegaConf.register_new_resolver("len", lambda x: len(x), replace=True)
    OmegaConf.register_new_resolver("eval", lambda x: eval(x), replace=True)
