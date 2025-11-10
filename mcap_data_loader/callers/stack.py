from mcap_data_loader.datasets.mcap_dataset import SampleStamped
from mcap_data_loader.utils.basic import float_range
from typing import Tuple, List, Dict, Union, Literal
from collections import ChainMap
from pydantic import BaseModel, PositiveInt, ConfigDict
from mcap_data_loader.utils.array_like import (
    get_device_auto,
    get_ns_name_by_array,
    get_namespace_by_name,
    dtype_equal,
    get_default_dtype,
    get_default_device,
    Array,
    NDArray,
    Tensor,
)
from mcap_data_loader.callers.basis import CallerBasis
from threading import Lock


NormStackValue = List[List[str]]
StackType = Dict[
    str,
    Union[NormStackValue, List[str], Tuple[List[str], List[Union[float, PositiveInt]]]],
]
DictBatch = ChainMap[str, Union[Array, List[Array], int]]


class BatchStackerConfig(BaseModel):
    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    stack: StackType
    """Configuration for stacking keys."""
    dtype: Union[Literal["same"], str] = "same"
    """Data type for the stacked arrays. If `same`, keep the original dtype.
    If empty, use the default dtype of the backend."""
    device: Union[Literal["same", "auto"], str] = "auto"
    """Device to move the stacked arrays to. If `same`, keep the original device.
    If empty, use the default device of the backend. If `auto`, try to use a best compatible device."""
    backend_in: Literal["torch", "numpy", "auto"] = "auto"
    """The input data backend."""
    backend_out: Literal["torch", "numpy", "list", "same"] = "same"
    """The output data backend."""


class BatchStacker(CallerBasis):
    """A caller that stacks specified keys from batched samples."""

    config: BatchStackerConfig

    def on_configure(self):
        self.stack = self._normalize_stack_config(self.config.stack)
        keys_info = {}
        self.keys_to_stack = set()
        for cat_key, list_keys in self.stack.items():
            keys_info[cat_key] = {}
            col_num = len(list_keys[0])
            cur_keys = []
            for c in range(col_num):
                for r, keys in enumerate(list_keys):
                    keys_info[cat_key][keys[c]] = [c, r]
                    self.keys_to_stack.add(keys[c])
                    cur_keys.append(keys[c])
            if len(cur_keys) != len(keys_info[cat_key]):
                raise ValueError(
                    f"Duplicate keys found in stacking config for category '{cat_key}': {cur_keys}"
                )
        self.keys_info: Dict[str, dict] = keys_info
        self._first_call = True
        self._lock = Lock()
        return True

    def _determine_functions(self, backend_in: str, dtype_in, device_in):
        # input process
        self._xp_in = get_namespace_by_name(backend_in)
        self._dtype_in = dtype_in
        self._device_in = device_in
        # output process
        config = self.config
        backend_out = backend_in if config.backend_out == "same" else config.backend_out
        self._xp_out = get_namespace_by_name(backend_out)
        dtype_out = config.dtype
        if dtype_out == "same" or dtype_equal(dtype_out, dtype_in):
            dtype_out = None
        elif not dtype_out:
            dtype_out = get_default_dtype(backend_out)
        else:
            dtype_out = getattr(self._xp_out, dtype_out)
        self._dtype_out = dtype_out
        device_out = config.device
        if device_out == "same":
            device_out = str(device_in)
        elif not device_out:
            device_out = get_default_device(backend_out)
        else:
            device_out = "" if device_out == "auto" else device_out
            device_out = get_device_auto(backend_out, device_out)
        self._device_out = device_out
        # determine output conversion function
        if backend_in == backend_out:
            # TODO: for torch, we may still need to move to device?
            self.convert_func = lambda x: x
        elif backend_out == "list":
            self.convert_func = self._to_list
        elif backend_out == "numpy":
            self.convert_func = self._torch_to_np
        else:
            self.convert_func = self._np_to_torch

    def _normalize_stack_config(self, stack: StackType) -> Dict[str, NormStackValue]:
        def process_value(config):
            if isinstance(config, tuple):
                keys, prefixes = config
                if len(prefixes) == 3 and isinstance(prefixes[2], int):
                    # range style
                    start, stop, step = prefixes
                    prefixes = float_range(start, stop, step)
                return [[f"{p}{k}" for k in keys] for p in prefixes]
            else:
                first = config[0]
                if isinstance(first, str):
                    return [config]
                else:
                    return config

        return {k: process_value(v) for k, v in stack.items()}

    def _np_to_torch(self, array: NDArray) -> Tensor:
        # no need to check dtype here, as the empty_func already creates the correct dtype
        return self._xp_out.from_numpy(array).to(
            device=self._device_out, non_blocking=True
        )

    def _torch_to_device(self, tensor: Tensor) -> Tensor:
        return tensor.to(device=self._device_out, non_blocking=True)

    def _torch_to_np(self, tensor: Tensor) -> NDArray:
        return tensor.cpu().numpy()

    def _to_list(self, data: Union[NDArray, Tensor]) -> list:
        return data.tolist()

    def _reset_buffers(self, batch_size: int):
        batch_stack = {}
        for cat_key, shape in self._batch_stack_shape.items():
            batch_stack[cat_key] = self._xp_in.empty(
                (batch_size,) + shape, dtype=self._dtype_in, device=self._device_in
            )
        batch_list: Dict[str, list] = {}
        for key in self._keys_no_stack:
            batch_list[key] = []
        return batch_stack, batch_list

    def _init_info(self, first_sample: SampleStamped):
        with self._lock:
            if not self._first_call:
                return
            batch_stack_shape = {}
            for cat_key, list_keys in self.stack.items():
                first_row = list_keys[0]
                row_num = len(list_keys)
                c2slice = []
                bias = 0
                for key in first_row:
                    inc = first_sample[key]["data"].shape[-1]
                    c2slice.append((bias, bias + inc))
                    bias += inc
                batch_stack_shape[cat_key] = (
                    row_num,
                    *first_sample[key]["data"].shape[:-1],
                    bias,
                )
                for key, config in self.keys_info[cat_key].items():
                    config[0] = c2slice[config[0]]
            one_value = first_sample[key]["data"]
            backend_in = (
                self.config.backend_in
                if self.config.backend_in != "auto"
                else get_ns_name_by_array(one_value)
            )
            self._determine_functions(backend_in, one_value.dtype, one_value.device)
            self._batch_stack_shape = batch_stack_shape
            self._keys_no_stack = first_sample.keys() - self.keys_to_stack
            self._first_call = False

    def __call__(self, batched_samples: List[SampleStamped]) -> DictBatch:
        if self._first_call:
            self._init_info(batched_samples[0])
        # allocate memory
        batch_size = len(batched_samples)
        batch_stack, batch_list = self._reset_buffers(batch_size)
        # fill in data
        for i, sample in enumerate(batched_samples):
            for cat_key, keys_dict in self.keys_info.items():
                for key, config in keys_dict.items():
                    (start, stop), r = config
                    batch_stack[cat_key][i, r, ..., start:stop] = sample[key]["data"]
            for key in self._keys_no_stack:
                batch_list[key].append(sample[key]["data"])
        # stack and move to device
        # TODO: use multi-treaded pin_memory and use a new cuda stream to copy asynchronously
        # TODO: test the performance vs tensor-dict
        final_batched = {}
        for catkey, data in batch_stack.items():
            final_batched[catkey] = self.convert_func(data)
        # keep the remaining batched dict unstacked
        return ChainMap(final_batched, batch_list, {"batch_size": batch_size})
