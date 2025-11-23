"""Example of directly creating multiple data loaders from yaml configuration."""

if __name__ == "__main__":
    from omegaconf import OmegaConf
    from mcap_data_loader.utils.hydra_utils import init_hydra_config, hydra_instance
    from pathlib import Path
    from pprint import pprint
    from typing import Dict, Iterable
    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("data_loaders_example")
    dict_config = init_hydra_config(Path(__file__).parent / "config.yaml")
    pprint(OmegaConf.to_container(dict_config))
    logger.info("Instantiating configuration...")
    config = hydra_instance(OmegaConf.to_container(dict_config))
    print("\n", "*" * 20, "DataLoader Configuration", "*" * 20)
    pprint(OmegaConf.to_container(config))
    print("\n")

    # the DataLoaders instance is callable to get the inner
    # data loaders dict
    data_loaders: Dict[str, Iterable] = config["data_loaders"]()

    def get_first_batchs():
        batchs = {}
        for stage, data_loader in data_loaders.items():
            logger.info(f"Iterating data loader for stage: {stage}")
            for batch in data_loader:
                batchs[stage] = batch
                break
        return batchs

    pprint(get_first_batchs())
    print("\n")
    pprint(get_first_batchs())
    print("Done.")
