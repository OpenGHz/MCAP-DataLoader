# MCAP Data Loading Performance Guide

This document explains the current performance-related parameters and execution
logic of the MCAP training data path used by `mcap_lerobot_train`.

The focus is the current implementation in:

- [`mcap_data_loader/datasets/mcap_lerobot.py`](/home/haizhou/MCAP-DataLoader/mcap_data_loader/datasets/mcap_lerobot.py)
- [`mcap_data_loader/datasets/mcap_dataset.py`](/home/haizhou/MCAP-DataLoader/mcap_data_loader/datasets/mcap_dataset.py)
- [`mcap_data_loader/serialization/basis.py`](/home/haizhou/MCAP-DataLoader/mcap_data_loader/serialization/basis.py)


## Scope

This guide is about the data path used when you run:

```bash
mcap_lerobot_train -c configs/mujoco.yaml
```

It does not describe generic MCAP utilities in the repo. It only covers the
training adapter that exposes MCAP data in a LeRobot-compatible format.


## Quick Summary

The current MCAP training path does these things at runtime:

1. Open one or more MCAP dataset roots.
2. Build episode datasets for each root.
3. For each episode:
   - iterate low-dimensional topics
   - iterate image/video attachments
   - zip and merge them into synchronized per-step samples
   - build horizon windows
   - convert them into LeRobot-style tensors
4. Batch them with PyTorch DataLoader.

The main performance levers are therefore:

- how many cameras are decoded
- how much low-dimensional data is merged
- whether multiple workers are actually used
- how many episodes exist for worker-level sharding
- whether the dataset is repeatedly reconstructing the same stream


## Main Config Parameters

### Training-side DataLoader parameters

These live in the top level training config, for example
[`configs/mujoco.yaml`](/home/haizhou/MCAP-DataLoader/configs/mujoco.yaml).

#### `num_workers`

Meaning:

- number of PyTorch DataLoader worker processes

Current behavior:

- passed into the training DataLoader in the LeRobot training script
- for the MCAP adapter, worker splitting is currently **episode-level**

Relevant implementation:

- worker sharding logic: [`mcap_lerobot.py`](/home/haizhou/MCAP-DataLoader/mcap_data_loader/datasets/mcap_lerobot.py#L368)
- DataLoader wrapper: [`mcap_lerobot.py`](/home/haizhou/MCAP-DataLoader/mcap_data_loader/datasets/mcap_lerobot.py#L559)

Practical impact:

- `num_workers=0`: all decoding and merging happen in the main process
- `num_workers>0`: different workers receive different episodes
- if `num_episodes < num_workers`, extra workers may stay idle

Important limitation:

- current sharding is not sample-level, it is episode-level
- if your dataset effectively has one episode, increasing `num_workers` will not
  scale well


#### `batch_size`

Meaning:

- number of already-built samples per training batch

Practical impact:

- larger batch sizes reduce per-batch Python overhead
- but they do not reduce per-sample decode/merge cost
- GPU memory and model memory often limit this first


### MCAP adapter parameters

These live under the `mcap:` section of the training config.

#### `states`

Meaning:

- list of low-dimensional observation topics concatenated into
  `observation.state`

Implementation:

- stored in config: [`mcap_lerobot.py`](/home/haizhou/MCAP-DataLoader/mcap_data_loader/datasets/mcap_lerobot.py#L38)
- converted in `_sample_keys_tensor()`: [`mcap_lerobot.py`](/home/haizhou/MCAP-DataLoader/mcap_data_loader/datasets/mcap_lerobot.py#L345)

Performance impact:

- more state keys means more numpy concatenation and tensor conversion
- usually much cheaper than video decode


#### `actions`

Meaning:

- list of low-dimensional action topics concatenated into `action`

Implementation:

- stacked in `_stack_sample_keys()`: [`mcap_lerobot.py`](/home/haizhou/MCAP-DataLoader/mcap_data_loader/datasets/mcap_lerobot.py#L325)

Performance impact:

- similar to `states`
- usually not the dominant cost unless action tensors are very large


#### `images`

Meaning:

- list of image/video attachment names to expose as
  `observation.images.*`

Implementation:

- mapped to LeRobot camera keys in `__init__()`: [`mcap_lerobot.py`](/home/haizhou/MCAP-DataLoader/mcap_data_loader/datasets/mcap_lerobot.py#L166)
- materialized in `_stack_horizon_item()`: [`mcap_lerobot.py`](/home/haizhou/MCAP-DataLoader/mcap_data_loader/datasets/mcap_lerobot.py#L353)

Performance impact:

- this is usually the largest cost driver
- every added camera increases attachment handling, decode work, and batch memory
- throughput comparisons should always account for camera count


#### `prefetch_items`

Meaning:

- background-thread item prefetch count inside `McapLeRobotDataset`

Implementation:

- config field exists here: [`mcap_lerobot.py`](/home/haizhou/MCAP-DataLoader/mcap_data_loader/datasets/mcap_lerobot.py#L50)
- internal queue-based prefetch implementation exists here:
  [`mcap_lerobot.py`](/home/haizhou/MCAP-DataLoader/mcap_data_loader/datasets/mcap_lerobot.py#L394)

Current status:

- `_build_item_iter()` currently returns `_iter_items()` directly
- so `prefetch_items` is effectively **disabled in the current training path**

Relevant line:

- [`mcap_lerobot.py`](/home/haizhou/MCAP-DataLoader/mcap_data_loader/datasets/mcap_lerobot.py#L434)

Why:

- earlier profiling showed queue/lock overhead dominated any gain


#### `shuffle_episodes` and `shuffle_seed`

Meaning:

- shuffle episode order between passes

Implementation:

- handled in `_iter_items()`: [`mcap_lerobot.py`](/home/haizhou/MCAP-DataLoader/mcap_data_loader/datasets/mcap_lerobot.py#L375)

Performance impact:

- usually small direct performance impact
- may slightly affect cache locality or file access patterns


### Horizon parameters

The adapter internally uses `HorizonConfig`, and training derives
`future_num` from `policy.action_delta_indices`.

Implementation:

- config field: [`mcap_lerobot.py`](/home/haizhou/MCAP-DataLoader/mcap_data_loader/datasets/mcap_lerobot.py#L44)
- horizon iterator: [`mcap_lerobot.py`](/home/haizhou/MCAP-DataLoader/mcap_data_loader/datasets/mcap_lerobot.py#L267)

Performance impact:

- larger future windows increase action stacking work
- still usually much cheaper than decoding multiple cameras


## Dataset-Level Performance Behavior

### `cache_stream`

This exists at the lower MCAP dataset layer:

- config field: [`mcap_dataset.py`](/home/haizhou/MCAP-DataLoader/mcap_data_loader/datasets/mcap_dataset.py#L58)
- behavior in `read_stream()`: [`mcap_dataset.py`](/home/haizhou/MCAP-DataLoader/mcap_data_loader/datasets/mcap_dataset.py#L135)

Meaning:

- if enabled, one full sample stream from a `.mcap` file is materialized in memory
  and reused later

Important note:

- this may also keep decoded image data in memory
- it is therefore **not** enabled by default in the current training path
- it is useful for experiments, but not ideal for fair comparisons against
  LeRobot, which mainly caches decoder state rather than whole decoded streams


### Episode dataset object reuse

Current behavior:

- `McapFlatBuffersEpisodeDataset` caches `SampleDataset` objects by episode index

Implementation:

- cache field: [`mcap_dataset.py`](/home/haizhou/MCAP-DataLoader/mcap_data_loader/datasets/mcap_dataset.py#L225)
- reuse in `__getitem__()`: [`mcap_dataset.py`](/home/haizhou/MCAP-DataLoader/mcap_data_loader/datasets/mcap_dataset.py#L272)

Why it matters:

- avoids repeatedly rebuilding sample-dataset wrappers
- avoids repeated Pydantic construction for the same episode dataset


### Fork-safe runtime state reset

Current behavior:

- `McapLeRobotDataset` probes one sample during initialization to infer feature
  shapes
- after that probe, it explicitly clears live reader / iterator runtime state
  before PyTorch workers start

Implementation:

- reset after first-sample probe:
  [`mcap_lerobot.py`](/home/haizhou/MCAP-DataLoader/mcap_data_loader/datasets/mcap_lerobot.py#L177)
- dataset-level reset:
  [`mcap_lerobot.py`](/home/haizhou/MCAP-DataLoader/mcap_data_loader/datasets/mcap_lerobot.py#L223)
- sample-dataset reset:
  [`mcap_dataset.py`](/home/haizhou/MCAP-DataLoader/mcap_data_loader/datasets/mcap_dataset.py#L152)
- episode-dataset reset:
  [`mcap_dataset.py`](/home/haizhou/MCAP-DataLoader/mcap_data_loader/datasets/mcap_dataset.py#L285)

Why it matters:

- the first implementation could leave open MCAP readers or decoder state in the
  main process
- with `num_workers>0`, worker processes could inherit that partially initialized
  runtime state and stall before producing the first batch
- clearing runtime state keeps the metadata probe but forces each worker to open
  its own fresh reader/decoder state


## Video Decode Path

### Decode backend choice

The current training adapter always constructs MCAP datasets with:

- `backend="torchcodec"`
- `frame_format="rgb24"`
- `dimension_order="NCHW"`

Implementation:

- [`mcap_lerobot.py`](/home/haizhou/MCAP-DataLoader/mcap_data_loader/datasets/mcap_lerobot.py#L230)

Fallback:

- if `torchcodec` import fails at training launch, the CLI appends
  `--dataset.video_backend=pyav`
- see [`mcap_lerobot.py`](/home/haizhou/MCAP-DataLoader/mcap_data_loader/datasets/mcap_lerobot.py#L541)


### Attachment decode caching

At the MCAP reader layer, there is a cache for attachment decoders:

- cache field: [`basis.py`](/home/haizhou/MCAP-DataLoader/mcap_data_loader/serialization/basis.py#L24)
- attachment iteration: [`basis.py`](/home/haizhou/MCAP-DataLoader/mcap_data_loader/serialization/basis.py#L80)

Current behavior for `video/mp4` attachments:

- decoder objects are cached
- base timestamp is cached
- frames are still decoded during iteration

Important distinction:

- this is **decoder-state caching**
- it is **not full frame caching**

This is closer to LeRobot's approach than materializing every decoded frame in RAM.


## End-to-End Runtime Logic

The current training adapter roughly does this:

```text
McapLeRobotDataset.__iter__()
    -> _iter_items()
        -> choose episode order
        -> split episodes across workers if num_workers > 0
        -> open matching episode datasets from each root
        -> _iter_episode_horizon_items()
            -> _merge_episode_samples()
                -> sample_dataset.read_stream()
                    -> reader.iter_samples()
                        -> iter_message_samples()
                        -> iter_attachment_samples()
        -> _stack_horizon_item()
    -> DataLoader collate_fn()
        -> batch tensors
        -> convert image tensors from uint8 to float32 in [0, 1]
```

This explains where the main costs come from:

- repeated stream interpretation
- repeated topic + attachment merge
- video decode work
- horizon construction
- batch-time image conversion


## Worker Parallelism

### What is parallel today

Current parallelism is:

- PyTorch DataLoader workers
- each worker receives a subset of episodes

Implementation:

- worker split: [`mcap_lerobot.py`](/home/haizhou/MCAP-DataLoader/mcap_data_loader/datasets/mcap_lerobot.py#L381)

This means:

- the adapter can benefit from `num_workers > 0`
- but only when there are enough episodes to distribute


### How `num_workers` works right now

The current parallel model is:

```text
main training process
    |
    +--> DataLoader worker process 0 -> handles episode subset 0
    +--> DataLoader worker process 1 -> handles episode subset 1
    +--> DataLoader worker process 2 -> handles episode subset 2
    +--> DataLoader worker process 3 -> handles episode subset 3
```

More concretely:

1. The main training process creates the PyTorch DataLoader.
2. When `num_workers > 0`, PyTorch launches multiple worker **processes**.
3. Each worker gets its own dataset instance.
4. Inside each worker, `McapLeRobotDataset._iter_items()` checks `get_worker_info()`
   and keeps only that worker's episode subset.
5. Each worker independently:
   - opens MCAP episode datasets
   - decodes topics
   - decodes attachments
   - merges samples
   - builds horizon items
   - returns training items
6. The produced items are sent back to the main training process through PyTorch
   DataLoader inter-process queues.
7. The main process receives prefetched items/batches and feeds them into the
   model training step.

So yes, conceptually this is:

- worker processes prepare future training data
- data is transferred back to the main process through IPC
- the main process consumes ready-made batches while workers continue preparing
  later ones

This is why `num_workers` can help even though the model itself is not running in
those workers.


### Does each worker prepare a whole batch?

Roughly yes, but the exact unit is controlled by PyTorch DataLoader internals.

Important points:

- workers produce dataset items
- DataLoader assembles them into batches using `collate_fn`
- with multiple workers, PyTorch also prefetches future work
- in the current wrapper, `prefetch_factor=2` is filled in automatically when
  `num_workers > 0` and the user did not specify one

Relevant code:

- DataLoader wrapper: [`mcap_lerobot.py`](/home/haizhou/MCAP-DataLoader/mcap_data_loader/datasets/mcap_lerobot.py#L587)

So it is reasonable to think of the pipeline as:

- workers are not only computing the current batch
- they are also trying to stay ahead of the trainer by preparing upcoming data


### Why this helps

The MCAP adapter does non-trivial per-sample work:

- topic decode
- attachment/video decode
- topic + attachment merge
- horizon construction
- tensor conversion
- image conversion during collate

With `num_workers=0`, all of that happens in the main training process.

With `num_workers>0`, that work moves into multiple background worker processes,
which lets data preparation overlap with model execution.

This overlap is the main reason worker parallelism can increase training
throughput.


### Why the benefit is currently limited by episode count

Current splitting is **episode-level**, not sample-level.

That means:

- if there are many episodes, different workers can do useful work in parallel
- if there are very few episodes, some workers will be idle

For example:

- `num_workers=4`, `num_episodes=8`: good chance of useful parallelism
- `num_workers=4`, `num_episodes=1`: almost no useful parallelism

This is the key limitation of the current implementation.


### What is not parallel today

These parts are still effectively sequential inside one worker:

- reading one episode stream
- topic/attachment zipping within one episode
- horizon window construction within one episode

So if a dataset has very few episodes, worker scaling will be limited.


## Important Current Limitations

### Single-episode bottleneck

If the training set effectively contains one episode, current worker-level sharding
has little room to help.

Reason:

- the split unit is episode, not sample index

Consequence:

- you may set `num_workers=4`
- but only one worker may do substantial work


### `prefetch_items` is currently inactive

The queue-based internal prefetch implementation still exists, but it is not in the
active training path.

Reason:

- earlier profiling showed queue and lock overhead dominating


### `cache_stream` is experimental for performance testing

It can be useful to understand upper-bound throughput, but it changes the memory
tradeoff and may no longer reflect a LeRobot-like comparison.


## Practical Tuning Advice

### If you want a fair comparison with LeRobot

- keep `cache_stream=False`
- use the same number of cameras
- use similar batch size
- test with `num_workers=4`


### If your dataset has many episodes

Try:

- `num_workers=4`
- then `num_workers=8` if CPU and storage allow

Because current worker sharding is episode-level, more episodes means better
parallel scaling.


### If your dataset has very few episodes

Changing `num_workers` alone may not help much.

In that case, the next real optimization target is:

- sample-level or index-level sharding
- or pre-materializing a training-oriented format


### If throughput is dominated by cameras

The most effective sanity checks are:

- reduce camera count
- reduce resolution if allowed
- verify decode backend


## Recommended Reading

- LeRobot comparison: [`docs/lerobot_performance_analysis.md`](/home/haizhou/MCAP-DataLoader/docs/lerobot_performance_analysis.md)
- MCAP adapter code: [`mcap_lerobot.py`](/home/haizhou/MCAP-DataLoader/mcap_data_loader/datasets/mcap_lerobot.py)
- lower-level MCAP datasets: [`mcap_dataset.py`](/home/haizhou/MCAP-DataLoader/mcap_data_loader/datasets/mcap_dataset.py)
- attachment decode logic: [`basis.py`](/home/haizhou/MCAP-DataLoader/mcap_data_loader/serialization/basis.py)
