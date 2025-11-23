from hydra import compose, initialize
from omegaconf import OmegaConf

with initialize(version_base=None, config_path="."):
    cfg = compose(config_name="config")
    print(hash(cfg))
    print(dict(**cfg))
    print(list(cfg.model.keys()))  # 输出: ['name', 'layers', 'pretrained']
    print(OmegaConf.to_container(cfg))