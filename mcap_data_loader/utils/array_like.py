from array_api_compat import array_namespace  # noqa: F401
from pydantic import BaseModel, computed_field
from typing import Any, Type, Tuple, Literal, Union
from typing_extensions import Self, TYPE_CHECKING
import importlib


if TYPE_CHECKING:
    from numpy.typing import NDArray
    from torch import Tensor

    Array = Union[NDArray, Tensor]
else:
    from typing import MutableSequence

    Array = MutableSequence
    Tensor = Any
    NDArray = Any


try:
    import numpy as np
    import torch
except ImportError:
    pass

NameSpace = Union[Literal["torch", "numpy"], str]


class ArrayInfo(BaseModel, frozen=True):
    """Information about an array-like object."""

    arr_type: Type
    """The type of the array-like object."""
    dtype: Any
    """The data type of the array-like object."""
    shape: Tuple[int, ...]
    """The shape of the array-like object."""
    device: Any
    """The device of the array-like object."""

    @computed_field
    @property
    def type_name(self) -> str:
        """The name of the type of the array-like object."""
        return self.arr_type.__name__

    @classmethod
    def from_array(cls, array: Array) -> Self:
        """Create an ArrayInfo from an array-like object."""
        return cls(
            arr_type=type(array),
            dtype=array.dtype,
            shape=array.shape,
            device=array.device,
        )


def get_namespace_by_name(name: NameSpace):
    """Get the array namespace by name."""
    try:
        if TYPE_CHECKING:
            try:
                return np
            except Exception:
                return torch
        else:
            return importlib.import_module(f"array_api_compat.{name}")
    except ImportError as e:
        raise ValueError(f"Backend '{name}' is not available or not installed.") from e


def get_array_type_by_ns_name(name: NameSpace) -> Type:
    """Get the array type by name."""
    if name == "numpy":
        return np.ndarray
    elif name == "torch":
        return torch.Tensor
    else:
        return str


def get_ns_name_by_array(array: Array) -> NameSpace:
    """Get the namespace name by array-like object."""
    return type(array).__module__


def get_tensor_device_auto(device: str = "") -> str:
    """Get the tensor device automatically."""
    if device:
        return device
    return f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"


def get_device_auto(ns: NameSpace, device: str = "") -> str:
    """Get the device automatically."""
    if ns == "numpy":
        return "cpu"
    elif ns == "torch":
        return get_tensor_device_auto(device)
    else:
        raise ValueError(f"Unsupported namespace '{ns}' for device retrieval.")


def dtype_to_str(dtype: Any) -> str:
    """Convert a data type to its string representation."""
    if isinstance(dtype, str):
        return dtype
    try:
        return dtype.__name__
    except AttributeError:
        return str(dtype).split(".")[-1]


def dtype_equal(dtype1: Any, dtype2: Any) -> bool:
    """Compare two data types for equality."""
    return dtype_to_str(dtype1) == dtype_to_str(dtype2)


def get_default_dtype(ns: NameSpace) -> Any:
    """Get the default data type for the given namespace."""
    if ns == "numpy":
        return np.float64
    elif ns == "torch":
        return torch.float32
    else:
        raise ValueError(f"Unsupported namespace '{ns}' for default dtype retrieval.")


def get_default_device(ns: NameSpace) -> Any:
    """Get the default device for the given namespace."""
    if ns == "numpy":
        return "cpu"
    elif ns == "torch":
        return torch.get_default_device()
    else:
        raise ValueError(f"Unsupported namespace '{ns}' for default device retrieval.")


if __name__ == "__main__":
    arr_np = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    arr_torch = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)

    info_np = ArrayInfo.from_array(arr_np)
    info_torch = ArrayInfo.from_array(arr_torch)

    print("NumPy Array Info:", info_np)
    print("Torch Tensor Info:", info_torch)

    print("NumPy Namespace:", get_namespace_by_name("numpy"))
    print("Torch Namespace:", get_namespace_by_name("torch"))
    print("Array Type by Namespace Name (numpy):", get_array_type_by_ns_name("numpy"))
    print("Array Type by Namespace Name (torch):", get_array_type_by_ns_name("torch"))
    print("Namespace Name by Array (NumPy):", get_ns_name_by_array(arr_np))
    print("Namespace Name by Array (Torch):", get_ns_name_by_array(arr_torch))
