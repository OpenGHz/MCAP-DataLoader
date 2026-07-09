<div align="center">

<h1>MCAP Data Loader</h1>

[![PyPI](https://img.shields.io/pypi/v/mcap-data-loader)](https://pypi.org/project/mcap-data-loader/)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

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

You should have LeRobot installed in your environment to use this feature. You can install it from PyPI (0.4.3 is tested):

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

The lists of topics specified by `states` and `actions` will be loaded and concatenated to form the `observation.state` and `action` required by lerobot, serving as low-dimensional state and action inputs in the training data. Meanwhile, `images` will be appended to the `observation.images` field, using the first part of the name (e.g., `env_camera` in the example above) as a suffix for image input, such as `observation.images.env_camera`, for use during training.

#### Vision-language-action policies (pi0.5)

Vision-language-action policies such as pi0.5 need two things beyond ACT, both handled by the `mcap` section:

- **Language task.** Each sample must carry a language instruction. It is extracted per-episode from an MCAP metadata record (by default `task_info.task_description`). Set `task_source` to `metadata` (default), `config` (use the static `task` string), or `none` (disable, e.g. for ACT).
- **Quantile statistics.** pi0.5 normalizes state/action with quantiles, so `q01`/`q99` stats are required. Set `compute_quantiles: true` to compute them with one extra pass over the dataset. This is auto-enabled when the policy uses quantile normalization.

pi0.5 also requires a non-empty `states`. A minimal example (see `configs/pi05.yaml`):

```yaml
policy:
  type: pi05
  chunk_size: 50
  n_action_steps: 50

mcap:
  states:
    - /follow/arm/pose/position
    - /follow/arm/pose/orientation
  actions:
    - /lead/arm/pose/position
    - /lead/arm/pose/orientation
  images:
    - /env_camera/color/image_raw
  task_source: metadata          # metadata | config | none
  task_metadata_name: task_info
  task_field: task_description    # or task_description_zh
  task: "do the task"            # fallback when metadata is missing
  compute_quantiles: true
```

The action chunk length, image resize to 224, and state/action padding are handled inside the pi0.5 model, so no data-side change is needed for those. The first run downloads the PaliGemma tokenizer/weights from the Hugging Face hub.

#### Organizing processed data

For processed data, MCAP is better suited to creating a new file that contains only the processed topics, rather than appending processed data back into the original file. For an example of generating processed topics, see [Data Processing](#data-processing).

During training, you can specify both the original dataset directory and the processed dataset directory at the same time. MCAP Data Loader will merge them automatically at runtime, so they can be consumed as if they were read from a single dataset.

A typical configuration looks like this:

```yaml
dataset:
  root: data
  repo_id:
    - mujoco
    - mujoco_processed
  streaming: true
```

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

### Data Processing

For pose-topic post-processing, see [docs/poses.md](docs/poses.md).

The script [mcap_data_loader/scripts/data_process/poses.py](mcap_data_loader/scripts/data_process/poses.py) can be used to generate:

- relative pose topics with `_rela` suffix
- `rotation_6d` topics converted from quaternion pose topics

Example:

```bash
python mcap_data_loader/scripts/data_process/poses.py \
  data/example \
  --keys /follow/arm/pose/position /follow/arm/pose/orientation \
  --targets rela rotation_6d
```




## License

See [LICENSE](LICENSE).
