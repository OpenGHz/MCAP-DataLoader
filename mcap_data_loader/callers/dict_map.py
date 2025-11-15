from mcap_data_loader.utils.basic import DictDataStamped, DataStamped, T
from mcap_data_loader.callers.basis import CallerBasis
from mcap_data_loader.utils.array_like import (
    ArrayTransferConfig,
    ArrayTransferMixin,
    AllBackend,
    Array,
)
from typing import Set
from collections import defaultdict


class DictMapConfig(ArrayTransferConfig, frozen=True):
    """Configuration for DictMap caller."""

    backend_in: AllBackend = "auto"
    """The input data backend."""
    keys_include: Set[str] = set()
    """The keys to include for mapping. If empty, all keys are included."""
    keys_exclude: Set[str] = set()
    """The keys to exclude from mapping. Applied after keys_include."""
    include_unmapped: bool = False
    """Whether to include unmapped keys in the output dict."""
    replace: bool = False
    """Whether to replace the original data in the output dict. If False and `include_unmapped`
    is True, a copy of the original data is made before mapping which is much slower. If True,
    the unmapped keys are included even if `include_unmapped` is False."""


class DictMap(CallerBasis[DictDataStamped[T]]):
    """A caller that maps the input dict data to another dict data according to the given mapping.
    This is mainly used to convert heterogeneous dict data to homogeneous dict data.
    """

    def __init__(self, config: DictMapConfig):
        self.config = config
        self._first_call = True
        self._transfers = []
        self._trans_keys = defaultdict(list)
        self._mixin_cls = ArrayTransferMixin
        self._include = config.include_unmapped
        self._replace = config.replace

    def __call__(self, data: DictDataStamped) -> DictDataStamped[T]:
        if self._first_call:
            config = self.config
            # determine keys to map
            keys = config.keys_include if config.keys_include else data.keys()
            keys = keys - config.keys_exclude
            self._no_map_keys = data.keys() - keys
            # TODO: should set conversion methods for each keys separately?
            for key in keys:
                stamped = data[key]
                value = stamped["data"]
                backend_in = config.backend_in
                if backend_in == "auto":
                    if isinstance(value, (list, tuple)):
                        backend_in = "list"
                    else:
                        backend_in = self._mixin_cls.get_backend_name(backend_in, value)
                backend_out = self._mixin_cls.get_backend_out(
                    backend_in, config.backend_out
                )
                if backend_in == "list":
                    dtype_in = None
                    device_in = "cpu"
                else:
                    value: Array
                    dtype_in = value.dtype
                    device_in = value.device
                kind = (backend_in, dtype_in, device_in, backend_out)
                self._trans_keys[kind].append(key)
            for kind, keys in self._trans_keys.items():
                backend_in, dtype_in, device_in, backend_out = kind
                mixin = self._mixin_cls()
                if backend_out != "list":
                    mixin._init_dtype_out(backend_out, dtype_in, config.dtype)
                    mixin._init_device_out(backend_out, device_in, config.device)
                if backend_out != "list":
                    mixin._init_xp_out(backend_out)
                if backend_in == "list":
                    if backend_in == backend_out:
                        convert_func = mixin.pass_through
                    else:
                        convert_func = mixin.list_to_output
                else:
                    mixin._init_in(backend_in, dtype_in, device_in)
                    convert_func = mixin.get_convert_func(backend_in, backend_out)
                self._transfers.append((keys, convert_func))
            self._first_call = False
        transfered = data if self._replace else data.copy() if self._include else {}
        for keys, convert_func in self._transfers:
            DataStamped.map_dict(data, convert_func, keys, transfered)
        return transfered


if __name__ == "__main__":
    import time
    import numpy as np

    input_dict = {
        "a": {"data": [1, 2, 3], "t": 0.0},
        "b": {"data": np.array([4, 5, 6]), "t": 0.0},
    }
    for backend_out in ("same", "torch", "numpy", "list"):
        print(f"--- backend_out: {backend_out} ---")
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
            f"mapped: {mapped}, time: {(end - start) * 1000:.3f} ms, warm up time: {warm_up_time * 1000:.3f} ms"
        )
