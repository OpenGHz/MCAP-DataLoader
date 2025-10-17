from mcap_data_loader.serialization.flb import (
    McapFlatBuffersWriter,
    FlatBuffersSchemas,
)
from mcap_data_loader.utils.mcap_utils import McapCLI
from time import time_ns


file_path = "data/written/0.mcap"

writer = McapFlatBuffersWriter()
writer.create_writer(file_path, overwrite=True)
writer.register_channel("/image/features", FlatBuffersSchemas.FLOAT_ARRAY)
writer.add_array("/image/features", [1.0, 2.0, 3.0], time_ns(), time_ns())
writer.unset_writer(finish=True)

# Verify the written file
mcap_cli = McapCLI("INFO")
assert not mcap_cli.is_mcap_corrupted(file_path)
# Show the file info
output = mcap_cli.run_command(f"info {file_path}")
print(mcap_cli.check_cmd_output(output))
