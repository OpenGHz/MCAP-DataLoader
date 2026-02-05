from mcap_data_loader.datasets.mcap_dataset import SampleStamped
from mcap_data_loader.utils.basic import float_range, ListMapping
from mcap_data_loader.utils.array_like import (
    Array,
    ArrayTransferMixin,
    ArrayTransferConfig,
)
from mcap_data_loader.callers.basis import CallerBasis
from mcap_data_loader.pipelines.horizon import HorizonItem, HorizonElement
from typing import Tuple, List, Dict, Union, Annotated
from pydantic import PositiveInt, AfterValidator, ConfigDict
from threading import Lock
from collections import defaultdict


DictBatch = Dict[str, Union[Array, List[Array], int]]
NormStackValue = List[List[str]]
StackTypeRaw = Dict[
    str,
    Union[
        NormStackValue,
        ListMapping[str],
        Tuple[List[str], List[Union[float, PositiveInt]]],
    ],
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


KeyDictType = Dict[str, Union[List[str], str]]


class HorizonStackerConfig(ArrayTransferConfig):
    """Configuration for HorizonStacker caller.
    The key of the stack dict is the new key for the stacked result, and the value specifies the keys to stack for that new key. If the value is empty, the key of the dict will be treated as the original key and no concatenating will be performed. If the value is a single string, it will be mapped to the target key and no concatenating will be
    performed."""

    past: KeyDictType = {}
    """Keys for past horizon to stack."""
    future: KeyDictType = {}
    """Keys for future horizon to stack."""
    now: KeyDictType = {}
    """Keys for current step to stack."""


class HorizonStacker(CallerBasis[Dict], ArrayTransferMixin):
    """A caller that stacks specified keys from horizon tuple."""

    def __init__(self, config: HorizonStackerConfig):
        self.config = config
        self._key_dicts = {
            "past": self.config.past,
            "future": self.config.future,
            "now": self.config.now,
        }
        self._process_keys()
        self._first_call = True
        self._lock = Lock()
        self._one_key = ""

    def _process_keys(self):
        self._ori_keys = defaultdict(dict)
        for kind, key_dict in self._key_dicts.items():
            for new_key in list(key_dict.keys()):
                keys = key_dict[new_key]
                if isinstance(keys, str) or not keys:
                    key_dict.pop(new_key)
                    self._ori_keys[kind][new_key] = new_key if not keys else keys

    def _check_keys(self, available_keys: set):
        for kind, key_dict in self._key_dicts.items():
            for new_key, keys in key_dict.items():
                keys = keys or [new_key]
                for key in keys:
                    if key not in available_keys:
                        raise KeyError(
                            f"Key '{key}' not found in available {kind} keys: {available_keys}."
                        )
        for kind, key_dict in self._ori_keys.items():
            for key in key_dict.values():
                if key not in available_keys:
                    raise KeyError(
                        f"Key '{key}' not found in available {kind} keys: {available_keys}."
                    )

    def _init_info(self, first_sample: HorizonItem[SampleStamped]):
        with self._lock:
            if not self._first_call:
                return
            first_dict = first_sample[0][0]
            self._check_keys(set(first_dict.keys()))
            self._one_key, one_value = next(iter(first_dict.items()))
            self._determine_from_array(
                one_value["data"],
                self.config.backend_out,
                self.config.dtype,
                self.config.device,
                self.config.backend_in,
            )
            self._first_call = False

    def _concat_stack(
        self,
        key_dict: KeyDictType,
        horizon_elem: HorizonElement,
        stacked_item: Dict[str, list],
    ):
        for new_key, keys in key_dict.items():
            stacked_item[new_key] = []
            for sample in horizon_elem:
                stacked_item[new_key].append(
                    self._xp_in.concatenate(
                        [sample[key]["data"] for key in keys], axis=-1
                    )
                )
            stacked_item[new_key] = self.convert_func(
                self._xp_in.stack(stacked_item[new_key])
            )

    def _stack(
        self,
        keys: Dict[str, str],
        horizon_elem: HorizonElement,
        stacked_item: Dict[str, list],
    ):
        for new_key, key in keys.items():
            stacked_item[new_key] = self.convert_func(
                self._xp_in.stack([sample[key]["data"] for sample in horizon_elem])
            )

    def __call__(self, horizon_item: HorizonItem[SampleStamped]):
        if self._first_call:
            self._init_info(horizon_item)
        past, future = horizon_item
        stacked_item = {}
        self._concat_stack(self._key_dicts["past"], past, stacked_item)
        self._concat_stack(self._key_dicts["future"], future, stacked_item)
        cur_data = horizon_item[0][-1]
        for new_key, keys in self._key_dicts["now"].items():
            stacked_item[new_key] = self.convert_func(
                self._xp_in.concatenate(
                    [cur_data[key]["data"] for key in keys], axis=-1
                )
            )
        self._stack(self._ori_keys["past"], past, stacked_item)
        self._stack(self._ori_keys["future"], future, stacked_item)
        for new_key, key in self._ori_keys["now"].items():
            stacked_item[new_key] = self.convert_func(cur_data[key]["data"])
        stacked_item["timestamp"] = cur_data[self._one_key]["t"]
        return stacked_item
