from mcap_data_loader.serialization.flb import McapFlatBuffersReader
from mcap_data_loader.utils.mcap_utils import McapCLI, McapTool
from pathlib import Path
import argparse


parser = argparse.ArgumentParser(description="Add topic statistics to MCAP files.")
parser.add_argument("paths", nargs="+", help="List of MCAP files/folders to process.")
parser.add_argument("-d", "--debug", action="store_true", help="Enable debug logging.")
args = parser.parse_args()

cli = McapCLI("DEBUG" if args.debug else "INFO")
for path in args.paths:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist.")
    if path.is_dir():
        mcap_files = list(path.rglob("*.mcap"))
    else:
        mcap_files = [path]
    for file in mcap_files:
        with open(file, "rb") as f:
            reader = McapFlatBuffersReader(f)
            if reader.has_topic_statistics():
                print(f"Skipping {file}.")
                continue
            print(f"Adding to {file}...")
            cli.add_attachment(
                file,
                *McapTool.topic_statistics_attachment_args(reader.topic_statistics),
            )
print("Done.")
