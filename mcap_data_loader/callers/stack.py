from mcap_data_loader.datasets.mcap_dataset import SampleStamped
from mcap_data_loader.utils.basic import float_range, ListMapping
from typing import Tuple, List, Dict, Union, Annotated
from pydantic import PositiveInt, AfterValidator, ConfigDict
from mcap_data_loader.utils.array_like import (
    Array,
    ArrayTransferMixin,
    ArrayTransferConfig,
)
from mcap_data_loader.callers.basis import CallerBasis
from threading import Lock


DictBatch = Dict[str, Union[Array, List[Array], int]]
NormStackValue = List[List[str]]
StackTypeRaw = Dict[
    str,
    Union[NormStackValue, ListMapping[str], Tuple[List[str], List[Union[float, PositiveInt]]]],
]


def normalize_stack_config(stack: StackTypeRaw) -> Dict[str, NormStackValue]:
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


StackType = Annotated[StackTypeRaw, AfterValidator(normalize_stack_config)]


class BatchStackerConfig(ArrayTransferConfig):
    """Configuration for BatchStacker caller."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    stack: StackType
    """Configuration for stacking keys, normalized to a consistent format automatically.
    The concatenated keys cannot be the same as any keys that do not need to be concatenated."""


class BatchStacker(CallerBasis[DictBatch], ArrayTransferMixin):
    """A caller that stacks specified keys from batched samples."""

    def __init__(self, config: BatchStackerConfig):
        self._stack = config.stack
        self.config = config
        self._keys_to_stack = set()
        self._first_call = True
        self._lock = Lock()
        keys_info = {}
        for cat_key, list_keys in self._stack.items():
            keys_info[cat_key] = {}
            col_num = len(list_keys[0])
            cur_keys = []
            for c in range(col_num):
                for r, keys in enumerate(list_keys):
                    keys_info[cat_key][keys[c]] = [c, r]
                    self._keys_to_stack.add(keys[c])
                    cur_keys.append(keys[c])
            if len(cur_keys) != len(keys_info[cat_key]):
                raise ValueError(
                    f"Duplicate keys found in stacking config for category '{cat_key}': {cur_keys}"
                )
        self._keys_info: Dict[str, dict] = keys_info

    def _reset_buffers(self, batch_size: int):
        batch_stack: Dict[str, Union[list, Array]] = {}
        for key in self._keys_no_stack:
            batch_stack[key] = []
        for cat_key, shape in self._batch_stack_shape.items():
            batch_stack[cat_key] = self._xp_in.empty(
                (batch_size,) + shape, dtype=self._dtype_in, device=self._device_in
            )
        return batch_stack

    def _init_info(self, first_sample: SampleStamped):
        with self._lock:
            if not self._first_call:
                return
            batch_stack_shape = {}
            for cat_key, list_keys in self._stack.items():
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
                for key, config in self._keys_info[cat_key].items():
                    config[0] = c2slice[config[0]]
            one_value = first_sample[key]["data"]
            self._determine_from_array(
                one_value,
                self.config.backend_out,
                self.config.dtype,
                self.config.device,
                self.config.backend_in,
            )
            self._batch_stack_shape = batch_stack_shape
            self._keys_no_stack = first_sample.keys() - self._keys_to_stack
            self._first_call = False

    def __call__(self, batched_samples: List[SampleStamped]):
        if self._first_call:
            self._init_info(batched_samples[0])
        keys_info = self._keys_info
        keys_no_stack = self._keys_no_stack
        convert_func = self.convert_func
        # allocate memory
        batch_size = len(batched_samples)
        batch_stack = self._reset_buffers(batch_size)
        # fill in data
        for i, sample in enumerate(batched_samples):
            for cat_key, keys_dict in keys_info.items():
                for key, config in keys_dict.items():
                    (start, stop), r = config
                    batch_stack[cat_key][i, r, ..., start:stop] = sample[key]["data"]
            # keep the remaining batched dict unstacked
            for key in keys_no_stack:
                batch_stack[key].append(sample[key]["data"])
        # stack and move to device
        # TODO: use multi-treaded pin_memory and use a new cuda stream to copy asynchronously
        # TODO: test the performance vs tensor-dict
        for catkey in keys_info:
            batch_stack[catkey] = convert_func(batch_stack[catkey])
        batch_stack["batch_size"] = batch_size
        return batch_stack
