<div align="center">

<h1>MCAP Data Loader</h1>

[![PyPI](https://img.shields.io/pypi/v/mcap-data-loader)](https://pypi.org/project/mcap-data-loader/)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/github/license/OpenGHz/MCAP-DataLoader)](LICENSE)

A Python library for loading and processing MCAP data files in a way that is more suitable for machine learning and robotics training pipelines.

</div>

## Features

- Dataset-style APIs for iterating MCAP data as episodes/samples
- Built-in statistics utilities (dataset-level and episode-level)
- Convenient access to topics and attachments
- Integration CLI for training with LeRobot using MCAP as the dataset backend

## Installation

Install from PyPI:

```bash
pip install mcap-data-loader
```

Or install from source:

```bash
git clone https://github.com/OpenGHz/MCAP-DataLoader.git --depth 1
cd MCAP-DataLoader
pip install -e .
```

## Quickstart (basic usage)

A basic example showing how to load MCAP files from a directory, inspect statistics, and iterate through episodes/samples:

```python
from mcap_data_loader.datasets.mcap_dataset import (
    McapFlatBuffersEpisodeDataset,
    McapFlatBuffersEpisodeDatasetConfig,
)
from pprint import pprint

dataset = McapFlatBuffersEpisodeDataset(
    McapFlatBuffersEpisodeDatasetConfig(
        data_root="data/example",
        # keys typically include topic names and optional special fields (e.g. "log_stamps")
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

More examples and detailed usage can be found in the [examples](examples) directory.

## Integration with LeRobot training

MCAP Data Loader provides a CLI to train LeRobot models using MCAP data files. This allows you to use MCAP datasets directly as the training data source for LeRobot, without needing to convert them into a different format.

You should have LeRobot installed in your environment to use this feature. You can install it from PyPI:

```bash
pip install lerobot
```

### Train with an MCAP dataset

Run:

```bash
mcap_lerobot_train -c configs/config.yaml
```

Recommended: place your config file under a `configs/` directory in your current working directory.

#### Configuration reference

The top level is the standard LeRobot configuration, with an additional `mcap` section for MCAP dataset loading settings:

```yaml
batch_size: 2
num_workers: 1
policy:
  type: act
  push_to_hub: false
  chunk_size: 2
  n_action_steps: 2

dataset:
  root: data
  repo_id: example
  streaming: true

mcap:
  states:
    - /follow/arm/joint_state/position
    - /follow/eef/joint_state/position
  actions:
    - /lead/arm/pose/position
    - /lead/arm/pose/orientation
  images:
    - /env_camera/color/image_raw
```

其中`states`和`actions`指定的topic列表将被加载和分别拼接成lerobot需要的`observation.state`和`action`，作为训练数据中的低维状态和动作输入，而
`images`则会按名称中的第一部分，例如上述中的`env_camera`，作为图像输入的后缀，添加到`observation.images`字段下，例如`observation.images.env_camera`，以供训练时使用。

Notes:
- `dataset.root` and `dataset.repo_id` are reused to specify the MCAP dataset root directory and dataset name.
- Command-line overrides compatible with LeRobot are supported and take the highest priority (they override values in the config file). For example:
  ```bash
  mcap_lerobot_train -c configs/config.yaml --dataset.repo_id=example_task
  ```

### Train with LeRobot’s original dataset format

If you want to use LeRobot’s original data format (while still using this CLI), add `--ori`:

```bash
mcap_lerobot_train -c configs/ori.yaml --ori
```

Make sure the dataset path in your config points to the actual LeRobot dataset location.

### Help / supported CLI args

Show supported parameters:

```bash
mcap_lerobot_train -h
```

If the output is long, redirect to a file:

```bash
mcap_lerobot_train -h > lerobot_help.txt
```

## License

See [LICENSE](LICENSE).
