from mcap_data_loader.serialization.flb import (
    McapFlatBuffersWriter,
    FlatBuffersSchemas,
)
from mcap_data_loader.utils.mcap_utils import McapCLI, McapTool
from time import time_ns
from mcap_data_loader.datasets.mcap_dataset import (
    McapFlatBuffersSampleDataset,
    McapFlatBuffersSampleDatasetConfig,
)
from pprint import pprint


file_path = "data/written/0.mcap"

writer = McapFlatBuffersWriter()
writer.create_writer(file_path, overwrite=True)
writer.register_channel("/image/features", FlatBuffersSchemas.FLOAT_ARRAY)
writer.add_float_array("/image/features", [1.0, 2.0, 3.0], time_ns(), time_ns())
writer.add_float_array("/image/features", [2.0, 3.0, 4.0], time_ns(), time_ns())
tool = McapTool(writer.get_writer())
tool.add_topic_statistics_attachment(writer.topic_statistics)
writer.unset_writer(finish=True)

# Verify the written file
mcap_cli = McapCLI("INFO")
assert not mcap_cli.is_mcap_corrupted(file_path)
# Show the file info
output = mcap_cli.run_command(f"info {file_path}")
pprint(mcap_cli.check_cmd_output(output))

dataset = McapFlatBuffersSampleDataset(
    McapFlatBuffersSampleDatasetConfig(data_root=file_path, keys=["/image/features"])
)
print("Statistics:")
pprint(dataset.statistics())
print("Samples:")
for sample in dataset:
    pprint(sample)
