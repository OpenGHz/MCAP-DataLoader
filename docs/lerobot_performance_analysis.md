# Why LeRobot Trains Faster Than The Current MCAP Adapter

This note summarizes why the original LeRobot dataset path is much faster than the
current `mcap-lerobot-train` path, even when MCAP uses fewer cameras.

It is based on:

- the current MCAP adapter implementation in this repo
- the local LeRobot source tree under `/DATA/disk1/haizhou/lerobot/src/lerobot`
- the profiling results collected during debugging


## Executive Summary

The main reason LeRobot is faster is not a single decoder call. It is that LeRobot
uses a training-oriented storage/layout, while the MCAP path still reconstructs
training samples on the fly.

LeRobot is faster because it:

- stores low-dimensional signals in a training-oriented format that can be read with very little online reconstruction
- stores videos as standalone files and decodes them in a way that is decoupled from MCAP message stream reconstruction
- avoids rebuilding episode/sample structures every epoch
- pushes less Python-side merging, zipping, and schema validation into the hot path

The MCAP path is slower because it still does these things at training time:

- iterates MCAP message streams every epoch
- merges multiple roots/topics into one sample stream on the fly
- decodes camera attachments while iterating samples
- constructs horizon windows in Python for every pass
- performs more per-epoch object construction and per-sample bookkeeping


## LeRobot Fast Path

### 1. Data is already reorganized for training

LeRobot stores:

- metadata in `meta/`
- low-dimensional data in chunked parquet files under `data/`
- videos as regular files under `videos/`

See the dataset structure described in [`lerobot_dataset.py`]( /DATA/disk1/haizhou/lerobot/src/lerobot/datasets/lerobot_dataset.py ) near the dataset class docstring.

This matters because the training loop is not reconstructing synchronized robot
samples from a logging format. It is reading from a training format directly.


### 2. Low-dimensional tensors do not need to be reconstructed from a logging stream

LeRobot's `DatasetReader` loads parquet into a Hugging Face dataset and sets a
transform to torch:

- `_load_hf_dataset()` in [`dataset_reader.py`]( /DATA/disk1/haizhou/lerobot/src/lerobot/datasets/dataset_reader.py )

Important properties:

- no need to scan MCAP messages to reconstruct the current frame
- no per-epoch topic merge step

The main point here is not that low-dimensional data is random-access by itself.
The main point is that LeRobot has already transformed robot logs into a format
where training reads tensors directly instead of repeatedly reinterpreting a log
stream.


### 3. Video decoding is query-based, not stream-joined

For image inputs, LeRobot computes the exact timestamps it needs, then queries only
those video frames:

- `_get_query_timestamps()` in [`dataset_reader.py`]( /DATA/disk1/haizhou/lerobot/src/lerobot/datasets/dataset_reader.py )
- `_query_videos()` in [`dataset_reader.py`]( /DATA/disk1/haizhou/lerobot/src/lerobot/datasets/dataset_reader.py )
- `decode_video_frames()` in [`video_utils.py`]( /DATA/disk1/haizhou/lerobot/src/lerobot/datasets/video_utils.py )

That is very different from the current MCAP path, where camera attachments are
zipped into the sample iterator and consumed frame-by-frame as part of the main
sample stream.


### 4. The key difference is not "map-style vs iterable" by itself

LeRobot training uses `LeRobotDataset`, and the non-streaming path is indeed a
regular `torch.utils.data.Dataset`:

- `class LeRobotDataset(torch.utils.data.Dataset)` in [`lerobot_dataset.py`]( /DATA/disk1/haizhou/lerobot/src/lerobot/datasets/lerobot_dataset.py )

But this should not be over-interpreted. In this comparison, `map-style` is not
the core explanation for the speed gap.

Why this is not the main point:

- a sequential streaming loader can also be very fast
- video still needs runtime decode in LeRobot
- low-dimensional iteration/decode is usually cheap enough that it should not
  explain a 3x gap by itself

So the more useful explanation is:

- LeRobot does less online reconstruction work per training sample
- the MCAP adapter still performs more "interpret the log format" work inside the
  training loop


## Current MCAP Slow Path

### 1. MCAP is still treated like a logging format during training

The MCAP adapter builds samples by iterating MCAP streams:

- `McapLeRobotDataset._iter_items()` in [`mcap_lerobot.py`]( /home/haizhou/MCAP-DataLoader/mcap_data_loader/datasets/mcap_lerobot.py )
- `McapLeRobotDataset._merge_episode_samples()` in [`mcap_lerobot.py`]( /home/haizhou/MCAP-DataLoader/mcap_data_loader/datasets/mcap_lerobot.py )
- `McapReaderBasis.iter_samples()` in [`basis.py`]( /home/haizhou/MCAP-DataLoader/mcap_data_loader/serialization/basis.py )

This means training time still includes work that LeRobot has already paid during
dataset creation/conversion.


### 2. Camera data is still interpreted as part of the MCAP log stream

In the current MCAP path, image attachments are iterated together with topic data:

- `iter_attachment_samples()` in [`basis.py`]( /home/haizhou/MCAP-DataLoader/mcap_data_loader/serialization/basis.py )

So every epoch still performs:

- attachment lookup
- video frame decode
- topic/attachment zip
- sample merge

Even if the decode backend itself is not dramatically slower, this still couples
camera decode to the process of reconstructing samples from the logging format.
That coupling is one of the main differences from LeRobot.


### 3. Horizon windows and merged samples are rebuilt every pass

The action horizon is assembled during iteration:

- `_iter_episode_horizon_items()` in [`mcap_lerobot.py`]( /home/haizhou/MCAP-DataLoader/mcap_data_loader/datasets/mcap_lerobot.py )

This alone is not likely to explain the whole gap, but it is part of the same
pattern: repeated fixed-dataset work that LeRobot largely avoids.


### 4. The adapter is still dominated by repeated fixed-dataset work

The profile results during debugging consistently showed large time in:

- `_iter_items`
- `_merge_episode_samples`
- `iter_samples`
- `iter_attachment_samples`
- frame decode paths

The key pattern is that the same dataset is being re-iterated and partially
reconstructed every epoch.


## What The Profiles Say

During profiling, the largest costs were not model-side. They were data-side:

- camera frame decode and image conversion
- MCAP sample iteration
- attachment iteration
- queue/thread overhead from prefetching
- repeated sample dataset construction before caching was added

This is consistent with the architecture difference:

- LeRobot pays more preprocessing cost before training
- the MCAP adapter pays more reconstruction cost during training


## Root Cause Comparison

### LeRobot

- Training format is already materialized.
- Numeric data does not need to be reassembled from message streams during training.
- Video access is separated from the process of rebuilding one synchronized sample stream from logs.
- The training loop mostly consumes already organized training data.

### MCAP adapter

- Logging format is reconstructed online.
- Numeric data is recovered by iterating messages, even if that part is individually fast.
- Video is consumed via attachment iteration tied to the same sample stream.
- More custom Python logic is executed to merge, synchronize, and rebuild samples every epoch.


## Data Path Diagram

The diagrams below focus on the video path and sample construction path, because
that is where the important architectural difference lives.

### LeRobot path

```text
Training loop
    |
    v
LeRobotDataset / StreamingLeRobotDataset
    |
    +--> low-dimensional data
    |       |
    |       v
    |   parquet / HF dataset
    |       |
    |       v
    |   ready-to-use rows / tensors
    |
    +--> image request for one sample / one timestamp set
            |
            v
        video file path lookup
            |
            v
        VideoDecoderCache
            |
            +--> cache hit: reuse decoder + file handle
            |
            +--> cache miss: create decoder once
            |
            v
        decode only requested frame indices
            |
            v
        return current sample frames
```

Key properties:

- low-dimensional data is already in a training-oriented format
- video decode is query-based
- decoder objects are cached
- decoded frames are usually not permanently cached
- sample construction does not require reconstructing a synchronized log stream

Important clarification:

- LeRobot does cache decoder state
- LeRobot does not normally cache a whole episode's decoded video frames in RAM

So an optimization that fully materializes decoded image streams in memory may be
useful as an experiment, but it should not be treated as an apples-to-apples
comparison against LeRobot training throughput.


### Current MCAP adapter path

```text
Training loop
    |
    v
McapLeRobotDataset
    |
    v
iterate episodes
    |
    v
open / reuse MCAP sample datasets
    |
    +--> iterate message samples
    |       |
    |       v
    |   decode low-dimensional topics from MCAP stream
    |
    +--> iterate attachment samples
            |
            v
        attachment lookup inside MCAP
            |
            v
        decoder cache
            |
            +--> cache hit: reuse decoder object
            |
            +--> cache miss: create decoder
            |
            v
        decode frames while iterating attachment stream
    |
    v
zip topics + attachments
    |
    v
merge into synchronized sample dict
    |
    v
build horizon window
    |
    v
stack tensors for the training batch
```

Key properties:

- low-dimensional data is still decoded from the logging stream at train time
- video decode is still tied to attachment iteration inside MCAP reconstruction
- decoder caching helps, but only at one layer
- topic merge + attachment merge + horizon construction still happen online
- the fixed offline dataset is repeatedly reinterpreted every epoch


### Where the main gap comes from

The important difference is not simply:

- "LeRobot caches video, MCAP does not"

That would be too strong and also inaccurate.

The more accurate difference is:

- LeRobot caches decoder state and reads from a dataset that is already organized
  for training
- MCAP currently caches some decoder state, but still reconstructs samples from a
  logging-oriented representation during training

So the expensive repeated work in MCAP is not only frame decode. It is the whole
chain:

- interpret topics
- interpret attachments
- join them
- rebuild synchronized per-step samples
- rebuild horizon structure

That is why decoder microbenchmarks alone do not explain the full training gap.


## Most Important Optimization Directions

These are the highest-value changes if the goal is to approach LeRobot speed.

### 1. Materialize episode-level training caches

Best option for fixed offline training:

- decode all camera frames for an episode once
- materialize merged topic + image samples once
- optionally materialize horizon items once

This directly attacks repeated work across epochs.


### 2. Move closer to a map-style dataset

This is a secondary optimization direction, not the main explanation for the
current gap.

The more important target is not `map-style` itself. The more important target is
to reduce how much log interpretation still happens during training.

If a future implementation becomes map-style as a side effect of materializing
episode caches, that can help. But it should be viewed as a consequence of better
data organization, not the main reason LeRobot is faster.


### 3. Separate "conversion" from "training"

The strongest architectural fix is to stop asking the training loop to interpret a
logging format.

In practice that means:

- convert MCAP once into a LeRobot-like intermediate format
- train from the converted format

This is the closest way to match original LeRobot performance, because it adopts the
same core idea: do dataset organization before training, not during training.


## Bottom Line

LeRobot is faster because it trains from a dataset that is already arranged for
training.

The current MCAP path is slower because it still performs sample reconstruction,
stream merging, attachment handling, and image decode orchestration inside the
training loop.

So the performance gap is fundamentally architectural:

- LeRobot is "read preprocessed training data"
- MCAP is still partly "interpret raw logged data at train time"

If we want MCAP training speed to approach LeRobot speed, the biggest wins will
come from reducing repeated reconstruction work:

- decode/cache camera data once per episode instead of redoing equivalent work every pass
- materialize merged per-step samples instead of rebuilding them from topics + attachments every epoch
- move more cost from "train time interpretation" to "pre-train conversion / caching"

That is the real performance center of gravity. `map-style` alone is not.
