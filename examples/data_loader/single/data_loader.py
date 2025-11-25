"""Example of creating a data loader using a YAML configuration based on Hydra."""

if __name__ == "__main__":
    from omegaconf import OmegaConf
    from mcap_data_loader.pipelines.basis import Pipeline, PipelineConfig
    from mcap_data_loader.datasets.mcap_dataset import (
        McapFlatBuffersEpisodeDataset,
        McapFlatBuffersEpisodeDatasetConfig,
    )
    from mcap_data_loader.utils.hydra_utils import init_hydra_config, hydra_instance
    from pprint import pprint
    from pathlib import Path
    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("data_loader_example")
    dict_config = init_hydra_config(Path(__file__).parent / "data_loader.yaml")
    pprint(OmegaConf.to_container(dict_config))
    logger.info("Instantiating configuration...")
    config = hydra_instance(dict_config)
    print("\n", "*" * 20, "Pipeline Configuration", "*" * 20)
    pprint(OmegaConf.to_container(config))
    print("\n")
    logger.info("Creating datasets...")
    root_dir = "data/example"
    data_root = root_dir
    keys = ["/follow/arm/joint_state/position", "/follow/eef/joint_state/position"]
    datasets = (
        McapFlatBuffersEpisodeDataset(
            McapFlatBuffersEpisodeDatasetConfig(data_root=data_root, keys=keys[:1])
        ),
        McapFlatBuffersEpisodeDataset(
            McapFlatBuffersEpisodeDatasetConfig(data_root=data_root, keys=keys[1:])
        ),
    )
    logger.info("Initializing the connector...")
    pipeline = Pipeline(PipelineConfig(pipeline=config))
    logger.info("Applying the pipeline...")
    data_loader = pipeline(datasets)
    logger.info(f"Iterating {data_loader}...")
    for sample in data_loader:
        pprint(sample)
        break
    logger.info("Done.")
