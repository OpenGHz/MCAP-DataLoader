from mcap_data_loader.serialization.ros.mcap import McapROSReader
import airdc
from pathlib import Path


path = Path(airdc.__file__).parent.parent / "data/aao_data/door_0/0.mcap"
assert path.exists(), f"Data file does not exist: {path}"

with open(path, "rb") as f:
    reader = McapROSReader(f)

    for sample in reader.iter_samples(["/robot/camera/env2/video_encoded"]):
        print(sample)
        break
