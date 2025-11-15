"""Data derivation"""

from mcap_data_loader.datasets.mcap_dataset import (
    McapFlatBuffersEpisodeDataset,
    McapFlatBuffersEpisodeDatasetConfig,
)
from mcap_data_loader.serialization.flb import McapFlatBuffersWriter, FlatBuffersSchemas
from mcap_data_loader.utils.mcap_utils import McapTool
from mcap.reader import make_reader
import time
from pathlib import Path
import uuid


base_data_root = "data/example"
derived_data_root = "data/derived"

dataset = McapFlatBuffersEpisodeDataset(
    McapFlatBuffersEpisodeDatasetConfig(data_root=base_data_root)
)

# file_sizes = {
#     f: os.path.getsize(f) / 1024**2
#     for d, files in dataset.all_files.items()
#     for f in files
# }

# # print(file_sizes)
# total_size = sum(file_sizes.values())
# print(f"Total size: {total_size:.2f} MB")
# start = time.perf_counter()
# file_hashes = dataset.all_file_hashes
# print(
#     f"Hashing perf: {(time.perf_counter() - start) / total_size * 1024:.2f} s/GB"
# )

uuid_hex = uuid.uuid1().hex
print(f"UUID1: {uuid_hex}")

flb_writer = McapFlatBuffersWriter()
for episode in dataset:
    episode_file_path = episode.config.data_root
    derived_file_path = Path(derived_data_root) / Path(episode_file_path).name
    print(f"Deriving data to {derived_file_path}...")
    flb_writer.create_writer(derived_file_path, overwrite=True)
    mcap_tool = McapTool(writer=flb_writer.get_writer())
    # mcap_tool.add_derive_metadata()
    flb_writer.register_channel("/image/features", FlatBuffersSchemas.FLOAT_ARRAY)
    flb_writer.add_float_array(
        "/image/features", [1.0, 2.0, 3.0], time.time_ns(), time.time_ns()
    )
    # file_hashes[episode_file_path]
    flb_writer.unset_writer(finish=True)
    stream = open(derived_file_path, "rb")
    mcap_tool.set_reader(make_reader(stream))


"""Data chain"""

#
{""}
