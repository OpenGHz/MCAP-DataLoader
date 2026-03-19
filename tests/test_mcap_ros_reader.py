from mcap_data_loader.serialization.ros import McapROSReader
import airdc
from pathlib import Path


path = Path(airdc.__file__).parent.parent / "data/ros/0.mcap"
assert path.exists(), f"Data file does not exist: {path}"

with open(path, "rb") as f:
    reader = McapROSReader(f)

    for sample in reader.iter_samples(["/robot/arm_right_lead/joint_states"]):
        print(sample)
        break
