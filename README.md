# MCAP Data Loader: A Python library for loading and processing MCAP data files in a way that is more suitable for machine learning.

## Installation

You can install the MCAP Data Loader using pip:

```bash
pip install mcap-data-loader
```

or from source:

```bash
git clone https://github.com/OpenGHz/MCAP-DataLoader.git --depth 1
cd MCAP-DataLoader
pip install -e .
```

## Usage

Here is an basic example showing how to load MCAP data files from a given directory, get statistics and iterate through the data:

```python
from mcap_data_loader.datasets.mcap_dataset import (
    McapFlatBuffersEpisodeDataset,
    McapFlatBuffersEpisodeDatasetConfig,
)
from pprint import pprint

dataset = McapFlatBuffersEpisodeDataset(
    McapFlatBuffersEpisodeDatasetConfig(
        data_root="data/example",
        keys=["/follow/arm/joint_state/position", "log_stamps"],
    )
)
print(f"All files: {dataset.all_files}")
print(f"Dataset length: {len(dataset)}")
print("Dataset statistics:")
pprint(dataset.statistics())
for episode in dataset:
    print(f"Current file: {episode.config.data_root}")
    for sample in episode:
        print(f"Sample keys: {sample.keys()}")
        break
    print(f"Episode length: {len(episode)}")
    print(f"All topics: {episode.reader.all_topic_names()}")
    print(f"All attachments: {episode.reader.all_attachment_names()}")
    print("Episode statistics:")
    pprint(episode.statistics())
    print("----" * 10)
```
For more examples and detailed usage, please refer to the [examples](examples) directory.


## Integration with lerobot training

The Mcap Data Loader provides a convenient way to train lerobot models using Mcap data files.
To train using the MCAP dataset, execute the following command (it is recommended to place the configuration file in the configs folder under the current working directory):

```bash
mcap_lerobot_train -c configs/config.yaml
```

Here is a configuration file reference (the top level contains LeRobot's configuration, with an additional mcap field for MCAP dataset loading-related settings):

```yaml
batch_size: 2
num_workers: 1
policy:
  type: act
  push_to_hub: false
  chunk_size: 2
  n_action_steps: 2
  # batch_size: 2
dataset:
  root: data
  repo_id: example
  streaming: true

mcap:
  states:
    - /follow/arm/joint_state/position
    - /follow/eef/joint_state/position
  images:
    - /env_camera/color/image_raw
  actions:
    - /lead/arm/pose/position
    - /lead/arm/pose/orientation
```

This reuses the root and repo_id fields under the original dataset configuration in LeRobot to specify the root directory and dataset name for the MCAP dataset, respectively.
Additionally, command-line parameter passing compatible with the original LeRobot is also supported. These have the highest priority and will override parameters in the configuration file. For example:

```bash
mcap_lerobot_train -c configs/config.yaml --dataset.repo_id=example_task
```

If you wish to use LeRobot's original data format and reuse the configuration file parameter passing functionality, simply add --ori to the command:

```bash
mcap_lerobot_train -c configs/ori.yaml --ori
```

Make sure to modify the dataset path in the configuration to the actual LeRobot dataset path.
For MCAP dataset training, the -c parameter is required, as the MCAP dataset loading configuration needs to be specified via the configuration file. For training with LeRobot data format, -c can be omitted, in which case the usage is identical to the original LeRobot.
Regardless of the data format, you can view the supported LeRobot command-line parameters by executing:

```bash
mcap_lerobot_train -h
```

Currently, the displayed parameters are numerous; you can redirect the output to a text file for easier viewing:

```bash
mcap_lerobot_train -h > lerobot_help.txt
```
