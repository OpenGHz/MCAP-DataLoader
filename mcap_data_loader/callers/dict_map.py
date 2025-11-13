from mcap_data_loader.utils.basic import DictDataStamped, T, DataStamped
from mcap_data_loader.callers.basis import CallerBasis
from mcap_data_loader.utils.array_like import (
    ArrayTransferConfig,
    AllBackend,
    ArrayTransferMixin,
    Array,
)
from typing import Set


class DictMapConfig(ArrayTransferConfig):
    """Configuration for DictMap caller."""

    backend_in: AllBackend = "auto"
    """The input data backend."""
    keys_include: Set[str] = set()
    """The keys to include for mapping. If empty, all keys are included."""
    keys_exclude: Set[str] = set()
    """The keys to exclude from mapping. Applied after keys_include."""


class DictMap(CallerBasis[DictDataStamped[T]], ArrayTransferMixin):
    """A caller that maps the input dict data to another dict data according to the given mapping."""

    def __init__(self, config: DictMapConfig):
        self.config = config
        self._first_call = True

    def __call__(self, data: DictDataStamped) -> DictDataStamped[T]:
        if self._first_call:
            config = self.config
            value = next(iter(data.values()))["data"]
            if config.backend_in == "auto":
                if isinstance(value, (list, tuple)):
                    backend_in = "list"
                else:
                    backend_in = self._get_backend_name(config.backend_in, value)
            backend_out = self._get_backend_out(backend_in, config.backend_out)

            if backend_in == "list":
                dtype_in = None
                device_in = "cpu"
            else:
                value: Array
                dtype_in = value.dtype
                device_in = value.device
            if backend_out != "list":
                self._init_dtype_out(backend_out, dtype_in, config.dtype)
                self._init_device_out(backend_out, device_in, config.device)
            if backend_out != "list":
                self._init_xp_out(backend_out)
            if backend_in == "list":
                if backend_in == backend_out:
                    self.convert_func = self._pass_through
                else:
                    self.convert_func = self._list_to_output
            else:
                self.convert_func = self._get_convert_func()
            keys = config.keys_include if config.keys_include else data.keys()
            self._keys = keys - config.keys_exclude
            self._first_call = False
        return DataStamped.map_dict(data, self.convert_func, self._keys)


if __name__ == "__main__":
    import time

    input_dict = {
        "a": {"data": [1, 2, 3], "t": 0.0},
        "b": {"data": [4, 5, 6], "t": 0.0},
    }
    for backend_out in ("same", "torch", "numpy", "list"):
        dict_map_config = DictMapConfig(backend_out=backend_out)
        dict_map = DictMap(dict_map_config)
        # warm up
        start = time.perf_counter()
        dict_map(input_dict)
        warm_up_time = time.perf_counter() - start
        # benchmark
        start = time.perf_counter()
        mapped = dict_map(input_dict)
        end = time.perf_counter()
        print(
            f"backend_out: {backend_out}, mapped: {mapped}, time: {(end - start) * 1000:.3f} ms, warm up time: {warm_up_time * 1000:.3f} ms"
        )
