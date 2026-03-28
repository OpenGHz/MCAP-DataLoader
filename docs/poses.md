# `poses.py` Usage

`mcap_data_loader/scripts/data_process/poses.py` is a small utility for generating derived pose topics from an MCAP dataset directory.

It currently supports:

- relative pose topics with `_rela` suffix
- `rotation_6d` topics converted from quaternion topics

The script writes processed episodes to a new output directory through `McapDataSampler`.

## Input

The script expects an MCAP episode directory such as:

```text
data/example/
  0.mcap
  1.mcap
  2.mcap
```

You can either:

- provide `--keys` explicitly
- omit `--keys` and let the script try to extract pose-related keys automatically

Supported pose-related suffixes currently include:

- `position`
- `orientation`
- `rotation_6d`
- `_rela`

## Output

By default, processed files are written to:

```text
<input_parent>/processed
```

You can also override it with `--out_dir`.

Derived topics include:

- `xxx_rela`
- `.../rotation_6d`
- `.../rotation_6d_rela`

## Examples

Generate both relative pose and `rotation_6d` topics:

```bash
python mcap_data_loader/scripts/data_process/poses.py \
  data/example \
  --keys /follow/arm/pose/position /follow/arm/pose/orientation \
  --targets rela rotation_6d
```

Write to a custom directory:

```bash
python mcap_data_loader/scripts/data_process/poses.py \
  data/example \
  --keys /follow/arm/pose/position /follow/arm/pose/orientation \
  --targets rela rotation_6d \
  --out_dir data/example_processed
```

Only generate `rotation_6d` topics:

```bash
python mcap_data_loader/scripts/data_process/poses.py \
  data/example \
  --keys /follow/arm/pose/orientation \
  --targets rotation_6d
```

## Arguments

- `path`: input MCAP dataset directory
- `--keys`: pose-related keys to process
- `--targets`: derived targets to generate, available values are `rela` and `rotation_6d`
- `--out_dir`: output directory

## Notes

- The script processes each episode independently.
- Relative values are computed using the first frame of each episode as reference.
- The script is intended for pose-like low-dimensional topics, not generic topic conversion.
