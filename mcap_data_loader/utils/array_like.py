from array_api_compat import array_namespace  # noqa: F401
from pydantic import BaseModel, computed_field
from typing import Any, Type, Tuple, Literal, Union, Optional
from typing_extensions import Self, TYPE_CHECKING
import importlib


if TYPE_CHECKING:
    from numpy.typing import NDArray
    from torch import Tensor

else:
    from typing import MutableSequence

    Tensor = Any
    NDArray = Any

Array = Union[NDArray, Tensor]


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

    @computed_field
    @property
    def ns(self) -> str:
        """The namespace name of the array-like object."""
        return self.arr_type.__module__

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


def get_tensor_device_auto(device: str = "auto") -> str:
    """Get the tensor device automatically.
    Args:
        device: The device string.
            If "auto", will try to use CUDA with the current device id if available.
            If empty, will use the default device of torch.
            Otherwise, will return the input device string as is.
    Returns:
        The device string.
    """
    if device == "auto":
        return (
            f"cuda:{torch.cuda.current_device()}"
            if torch.cuda.is_available()
            else "cpu"
        )
    elif device:
        return device
    else:
        return str(torch.get_default_device())


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


def convert_str_dtype(ns: str, dtype: str) -> Any:
    """Convert a string data type to the corresponding data type in the given namespace."""
    if ns == "numpy":
        return getattr(np, dtype)
    elif ns == "torch":
        return getattr(torch, dtype)
    else:
        raise ValueError(f"Unsupported namespace '{ns}' for dtype conversion.")


def tensor_to_ndarray(
    tensor: Tensor,
    transpose: Tuple[int, ...],
    scale: float = 1.0,
    dtype: Union[np.dtype, str] = None,
) -> NDArray:
    """Convert a torch Tensor to a numpy ndarray."""
    arr = tensor.cpu().numpy() * scale
    if transpose:
        arr = arr.transpose(*transpose)
    if dtype is not None:
        return arr.astype(getattr(np, dtype) if isinstance(dtype, str) else dtype)
    return arr


InputBackend = Literal["torch", "numpy", "auto"]
AllBackend = Union[InputBackend, Literal["list"]]


class ArrayTransferConfig(BaseModel, frozen=True):
    """Common configuration for ArrayTransfer."""

    dtype: Union[Literal["same"], str] = "same"
    """Data type for the output arrays. If `same`, keep the original dtype.
    If empty, use the default dtype of the backend."""
    device: Union[Literal["same", "auto"], str] = "auto"
    """Device to move the output arrays to. If `same`, keep the original device.
    If empty, use the default device of the backend. If `auto`, try to use a best compatible device."""
    backend_in: InputBackend = "auto"
    """The input data backend."""
    backend_out: Union[AllBackend, Literal["same"]] = "same"
    """The output data backend."""


class ArrayTransferMixin:
    """Utility class for transferring array-like objects between backends and devices."""

    @staticmethod
    def get_backend_name(backend: str, array: Optional[Array] = None) -> str:
        if backend == "auto":
            if array is None:
                raise ValueError("An array must be provided when backend is 'auto'.")
            return get_ns_name_by_array(array)
        return backend

    def _determine_functions(
        self,
        backend_in: str,
        dtype_in,
        device_in,
        backend_out: str,
        dtype_out: str,
        device_out: str,
        array: Optional[Array] = None,
    ):
        # input process
        backend_in = self.get_backend_name(backend_in, array)
        self._init_in(backend_in, dtype_in, device_in)
        # output process
        backend_out = self.get_backend_out(backend_in, backend_out)
        self._init_xp_out(backend_out)
        self._init_dtype_out(backend_out, dtype_in, dtype_out)
        self._init_device_out(backend_out, device_in, device_out)
        # determine output conversion function
        self.convert_func = self.get_convert_func(backend_in, backend_out)

    def _init_in(self, backend_in: str, dtype_in, device_in):
        self._xp_in = get_namespace_by_name(backend_in)
        self._dtype_in = dtype_in
        self._device_in = device_in

    def _init_dtype_out(self, backend_out: str, dtype_in: str, dtype_out: str):
        if dtype_out == "same" or dtype_in is None or dtype_equal(dtype_out, dtype_in):
            dtype_out = None
        elif not dtype_out:
            dtype_out = get_default_dtype(backend_out)
        else:
            dtype_out = getattr(self._xp_out, dtype_out)
        self._dtype_out = dtype_out

    def _init_device_out(self, backend_out: str, device_in: str, device_out: str):
        if device_out == "same":
            device_out = str(device_in) if device_in else "cpu"
        else:
            device_out = get_device_auto(backend_out, device_out)
        self._device_out = device_out

    def _determine_from_array(
        self,
        array: Array,
        backend_out: str,
        dtype_out: str,
        device_out: str,
        backend_in: str = "auto",
    ):
        self._determine_functions(
            backend_in,
            array.dtype,
            array.device,
            backend_out,
            dtype_out,
            device_out,
            array,
        )

    def get_convert_func(self, backend_in: str, backend_out: str):
        if backend_in == backend_out:
            if (
                self._dtype_in == self._dtype_out
                and self._device_in == self._device_out
            ):
                func = self.pass_through
            else:
                # TODO: for torch, use non_blocking=True
                func = self._self_convert
        elif backend_out == "list":
            func = self.to_list
        elif backend_out == "numpy":
            func = self.torch_to_np
        else:
            func = self._np_to_torch
        return func

    def list_to_output(self, data: list) -> Array:
        return self._xp_out.asarray(
            data, dtype=self._dtype_out, device=self._device_out
        )

    def _np_to_torch(self, array: NDArray) -> Tensor:
        # no need to check dtype here, as the empty_func already creates the correct dtype
        return self._xp_out.from_numpy(array).to(
            device=self._device_out, non_blocking=True
        )

    def _torch_to_device(self, tensor: Tensor) -> Tensor:
        return tensor.to(device=self._device_out, non_blocking=True)

    @staticmethod
    def torch_to_np(tensor: Tensor) -> NDArray:
        return tensor.cpu().numpy()

    @staticmethod
    def to_list(data: Union[NDArray, Tensor]) -> list:
        return data.tolist()

    def _self_convert(self, data: Array):
        return self._xp_out.astype(
            data, self._dtype_out, copy=True, device=self._device_out
        )

    @staticmethod
    def pass_through(data):
        return data

    @staticmethod
    def get_backend_out(backend_in: str, backend_out: str) -> str:
        return backend_in if backend_out == "same" else backend_out

    def _init_xp_out(self, backend_out: str):
        self._xp_out = get_namespace_by_name(backend_out)


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
