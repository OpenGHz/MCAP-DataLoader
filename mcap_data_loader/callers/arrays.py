from mcap_data_loader.utils.array_like import (
    Array,
    ArrayInfo,
    array_namespace,
)
from mcap_data_loader.callers.basis import CallerBasis


class ArrayCallerBasis(CallerBasis[Array]):
    """A caller that outputs array-like objects."""

    def __init__(self, config=None):
        self._first = True

    def _warm_up(self, output: Array) -> None:
        if 3 <= len(output.shape) <= 5:
            self.output_info = ArrayInfo.from_array(output)
            self.output_xp = array_namespace(output)
        else:
            raise ValueError(
                "The shape of the caller output must be (B, T, C, H, W), (B, T, H, W) or (B, T, D), "
                "i.e. (batch, time, channel, height, width), (batch, time, height, width) or "
                f"(batch, time, dimension). But got shape: {output.shape}"
            )
