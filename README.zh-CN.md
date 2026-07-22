<div align="center">

<h1>MCAP Data Loader</h1>

[![PyPI](https://img.shields.io/pypi/v/mcap-data-loader)](https://pypi.org/project/mcap-data-loader/)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

一个用于加载和处理 MCAP 数据文件的 Python 库，使其更适合机器学习与机器人训练流水线。

[English](README.md) | **简体中文**

</div>

## 功能特性

- 以数据集（Dataset）风格的 API 将 MCAP 数据按 episode/sample 迭代
- 内置统计工具（数据集级与 episode 级）
- 便捷访问 topic 与 attachment
- 提供以 MCAP 作为数据后端、用于 LeRobot 训练的集成 CLI

## 安装

从 PyPI 安装：

```bash
pip install mcap-data-loader
```

或从源码安装：

```bash
git clone https://github.com/OpenGHz/MCAP-DataLoader.git --depth 1
cd MCAP-DataLoader
pip install -e .
```

## 快速开始（基础用法）

下面的基础示例展示了如何从目录中加载 MCAP 文件、查看统计信息，并遍历 episode/sample：

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

更多示例和详细用法可参见 [examples](examples) 目录。

## 与 LeRobot 训练集成

MCAP Data Loader 提供了一个 CLI，用于直接使用 MCAP 数据文件训练 LeRobot 模型。这样你就可以把 MCAP 数据集直接作为 LeRobot 的训练数据源，而无需将其转换为其他格式。

使用该功能需要环境中已安装 LeRobot。你可以从 PyPI 安装（已在 0.4.3 上测试）：

```bash
pip install lerobot
```

### 使用 MCAP 数据集训练

运行：

```bash
mcap-lerobot-train -c configs/config.yaml
```

建议：将配置文件放在当前工作目录下的 `configs/` 目录中。

#### 配置参考

顶层是标准的 LeRobot 配置，额外增加了一个用于 MCAP 数据集加载设置的 `mcap` 小节：

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

`states` 与 `actions` 指定的 topic 列表会被加载并拼接，形成 LeRobot 所需的 `observation.state` 与 `action`，作为训练数据中的低维状态与动作输入。与此同时，`images` 会被追加到 `observation.images` 字段中，并使用名称的第一段（例如上面例子中的 `env_camera`）作为图像输入的后缀，例如 `observation.images.env_camera`，供训练时使用。

#### 视觉-语言-动作策略（pi0.5）

诸如 pi0.5 这样的视觉-语言-动作策略在 ACT 之外还需要两样东西，二者都由 `mcap` 小节处理：

- **语言任务（Language task）。** 每个 sample 必须携带一条语言指令。它会按 episode 从一条 MCAP 元数据记录中提取（默认取 `task_info.task_description`）。将 `task_source` 设为 `metadata`（默认）、`config`（使用静态的 `task` 字符串）或 `none`（禁用，例如用于 ACT）。
- **分位数统计（Quantile statistics）。** pi0.5 使用分位数对 state/action 归一化，因此需要 `q01`/`q99` 统计量。设置 `compute_quantiles: true` 会额外对数据集扫描一遍来计算它们。当策略使用分位数归一化时，此项会自动启用。

pi0.5 还要求 `states` 非空。一个最小示例（参见 `configs/pi05.yaml`）：

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

动作 chunk 长度、图像 resize 到 224，以及 state/action 的 padding 都在 pi0.5 模型内部处理，因此数据侧无需为这些做任何改动。首次运行会从 Hugging Face hub 下载 PaliGemma 的 tokenizer/权重。

#### 组织处理后的数据

对于处理后的数据，MCAP 更适合创建一个只包含处理后 topic 的新文件，而不是把处理结果追加回原始文件。生成处理后 topic 的示例见 [airdc `process_poses`](https://github.com/OpenGHz/data-collection/blob/develop/airdc/scripts/process_poses.py) 脚本。

在训练时，你可以同时指定原始数据集目录和处理后数据集目录。MCAP Data Loader 会在运行时自动合并它们，使其如同从单一数据集读取一样被使用。

一个典型配置如下：

```yaml
dataset:
  root: data
  repo_id:
    - mujoco
    - mujoco_processed
  streaming: true
```

说明：
- `dataset.root` 与 `dataset.repo_id` 被复用来指定 MCAP 数据集根目录与数据集名称。
- 支持与 LeRobot 兼容的命令行覆盖参数，且优先级最高（会覆盖配置文件中的值）。例如：
  ```bash
  mcap-lerobot-train -c configs/config.yaml --dataset.repo_id=example_task
  ```

### 使用 LeRobot 原生数据格式训练

如果你想使用 LeRobot 的原生数据格式（同时仍使用本 CLI），请加上 `--ori`：

```bash
mcap-lerobot-train -c configs/ori.yaml --ori
```

请确保配置中的数据集路径指向实际的 LeRobot 数据集位置。

### 帮助 / 支持的命令行参数

查看支持的参数：

```bash
mcap-lerobot-train -h
```

如果输出较长，可重定向到文件：

```bash
mcap-lerobot-train -h > lerobot_help.txt
```

### 数据处理

关于 pose topic 的后处理（生成相对 pose 与 `rotation_6d` topic），参见 [airdc](https://github.com/OpenGHz/data-collection) 项目中的 [`airdc.scripts.process_poses`](https://github.com/OpenGHz/data-collection/blob/develop/airdc/scripts/process_poses.py) 脚本。

该脚本可用于生成：

- 带 `_rela` 后缀的相对 pose topic
- 由四元数 pose topic 转换而来的 `rotation_6d` topic

示例：

```bash
python3 -m airdc.scripts.process_poses \
  data/example \
  --keys /follow/arm/pose/position /follow/arm/pose/orientation \
  --targets rela rotation_6d
```

## 文档

- [MCAP loader 性能](docs/mcap_loader_performance.md) —— MCAP 数据加载器的基准测试与调优说明
- [LeRobot 性能分析](docs/lerobot_performance_analysis.md) —— 使用 MCAP 进行 LeRobot 训练的吞吐分析

更多可运行的示例见 [examples](examples) 目录。

## 项目结构

```text
mcap_data_loader/
├── basis/          # Config-able base classes (datasets, loaders, data types)
├── callers/        # Composable transforms (map, normalize, stack, policy, ...)
├── configurers/    # Hydra / config wiring
├── data_types/     # Shared data-type definitions
├── datasets/       # MCAP dataset APIs + LeRobot training integration
├── pipelines/      # Data pipeline stages (horizon, flatten, merge, slice, ...)
├── schemas/        # FlatBuffers schemas (.fbs / .bfbs)
├── scripts/        # Data-processing / helper scripts
├── serialization/  # MCAP / ROS / FlatBuffers / video (de)serialization
└── utils/          # Shared utilities
```

## 贡献

欢迎贡献！请阅读 [CONTRIBUTING.md](CONTRIBUTING.md) 了解开发环境搭建、代码风格与 Pull Request 流程，并留意我们的[行为准则](CODE_OF_CONDUCT.md)。

## 获取帮助

- 提问与使用帮助：参见 [SUPPORT.md](SUPPORT.md)
- Bug 报告与功能请求：提交 [issue](https://github.com/OpenGHz/MCAP-DataLoader/issues)
- 安全问题报告：参见 [SECURITY.md](SECURITY.md)

## 许可证

本项目基于 [MIT 许可证](LICENSE) 授权。
