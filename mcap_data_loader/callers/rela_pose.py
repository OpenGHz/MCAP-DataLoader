from pydantic import BaseModel, ConfigDict
from typing import List, Literal, Any
from collections.abc import Callable
from mcap_data_loader.callers.basis import CallerBasis
from mcap_data_loader.datasets.mcap_dataset import SampleStamped
from mcap_data_loader.utils.rela_abs import to_relative_pose


class RelaPoseConfig(BaseModel, frozen=True):
    """Configuration for RelaPose caller."""

    model_config = ConfigDict(extra="forbid")

    positions: List[str]
    """The keys to compute the relative position (xyz) for."""
    orientations: List[str]
    """The keys to compute the relative orientation for."""
    reference: Callable[[Any], bool]
    """A function that takes in the data and returns a boolean mask indicating which samples to use as reference for relative pose computation."""
    euler_mode: Literal["xyz"] = "xyz"
    """The Euler angle convention to use for orientation computation."""
    quaternion_mode: Literal["xyzw", "wxyz"] = "xyzw"
    """The quaternion convention to use for orientation computation."""
    rot6d: bool = False
    """Whether to compute the 6D rotation representation instead of quaternion."""

    def model_post_init(self, context):
        if len(self.positions) != len(self.orientations):
            raise ValueError(
                f"Length of positions and orientations must be the same, but got {len(self.positions)} and {len(self.orientations)}"
            )


class RelaPose(CallerBasis[SampleStamped]):
    """Compute the relative pose (position and orientation) for the specified keys."""

    def __init__(self, config: RelaPoseConfig):
        self.config = config
        self._ref_position = {}
        self._ref_orientation = {}

    def __call__(self, data: SampleStamped):
        config = self.config
        is_ref = config.reference(data)
        if is_ref:
            for pos_key, ori_key in zip(config.positions, config.orientations):
                self._ref_position[pos_key] = data[pos_key]["data"]
                self._ref_orientation[ori_key] = data[ori_key]["data"]
        for pos_key, ori_key in zip(self._ref_position, self._ref_orientation):
            data[pos_key]["data"], quat, rot6d = to_relative_pose(
                self._ref_position[pos_key],
                self._ref_orientation[ori_key],
                data[pos_key]["data"],
                data[ori_key]["data"],
            )
            data[ori_key]["data"] = rot6d if config.rot6d else quat
        return data


if __name__ == "__main__":
    pass
