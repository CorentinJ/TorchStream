from abc import ABC
from typing import Generic, Tuple

import numpy as np
import torch
from numpy.typing import ArrayLike, DTypeLike

from torchstream.sequence.dtype import SeqArray, SeqArrayLike, SeqDTypeLike, dtypes_compatible, seqdtype, to_seqdtype


def _to_slice(idx):
    if isinstance(idx, slice):
        return idx
    return slice(idx, idx + 1)


class ArrayInterface(ABC, Generic[SeqArray]):
    dtype: seqdtype

    # TODO: overloads
    def __new__(cls, dtype_like: SeqDTypeLike | SeqArrayLike, device: str | torch.device = None):
        if cls is ArrayInterface:
            dtype = to_seqdtype(dtype_like)

            if isinstance(dtype, torch.dtype):
                cls = TensorInterface
            else:
                assert isinstance(dtype, np.dtype), "Internal error"
                cls = NumpyArrayInterface

        return object.__new__(cls)

    def get(self, arr: SeqArray, *idx) -> SeqArray:
        raise NotImplementedError()

    def get_along_dim(self, array: SeqArray, idx, dim: int) -> SeqArray:
        """
        Convenience method to index along a single dimension, returning the full space across all other dimensions.
        """
        slices = [slice(None)] * len(self.get_shape(array))
        slices[dim] = _to_slice(idx)
        return self.get(array, tuple(slices))

    def set(self, arr: SeqArray, *idx, value) -> None:
        raise NotImplementedError()

    def set_along_dim(self, array: SeqArray, idx, dim: int, value) -> None:
        """
        Convenience method to set values across a slice of a given dimension, including the full space across all other
        dimensions.
        FIXME: not suited for setting a single element atm
        """
        slices = [slice(None)] * len(self.get_shape(array))
        slices[dim] = _to_slice(idx)
        self.set(array, tuple(slices), value=value)

    def get_shape(self, arr: SeqArray) -> Tuple[int, ...]:
        raise NotImplementedError()

    def matches(self, arr: SeqArray | SeqDTypeLike) -> bool:
        """
        Returns whether the given array matches the specification of this interface.
        For the purpose of this class, matching means that the array is from the same library and has a numerical
        representation of the same kind (floating point, integer, complex, ...).
        """
        return dtypes_compatible(self.dtype, to_seqdtype(arr))

    def normalize(self, arr: SeqArrayLike) -> SeqArray:
        """
        Normalizes the given array to the interface's dtype. This is a no-op if the array already matches the interface.
        Normalizing may imply copying to a new container, casting to a different dtype, or changing the memory location.
        """
        raise NotImplementedError()

    def copy(self, arr: SeqArray) -> SeqArray:
        raise NotImplementedError()

    def concat(self, *arrays: SeqArray, dim: int) -> SeqArray:
        raise NotImplementedError()

    def new_empty(self, *shape: int | Tuple[int, ...]) -> SeqArray:
        """
        Returns an empty array of the given shape. The array's values are uninitialized.
        """
        raise NotImplementedError()

    def new_zeros(self, *shape: int | Tuple[int, ...]) -> SeqArray:
        """
        Returns an array of the given shape, filled with zeros.
        """
        return self.new_empty(*shape).zero_()

    def new_randn(self, *shape: int | Tuple[int, ...]) -> SeqArray:
        """
        Sample a sequence of the given size from a normal distribution (discretized for integer types).
        """
        raise NotImplementedError()


class NumpyArrayInterface(ArrayInterface[np.ndarray]):
    def __init__(self, dtype_like: DTypeLike | ArrayLike, device=None):
        assert not device
        # TODO: limit to numerical types (i.e. not strings)
        #   -> Why though? For the NaN trick?
        self.dtype = to_seqdtype(dtype_like)

    def get(self, arr: np.ndarray, *idx) -> np.ndarray:
        return arr.__getitem__(*idx)

    def set(self, arr: np.ndarray, *idx, value) -> None:
        arr.__setitem__(*idx, value)

    def get_shape(self, arr: np.ndarray) -> Tuple[int, ...]:
        return arr.shape

    def new_empty(self, *shape: int | Tuple[int, ...]) -> np.ndarray:
        return np.empty(shape[0] if isinstance(shape[0], tuple) else shape, dtype=self.dtype)

    def new_zeros(self, *shape: int | Tuple[int, ...]) -> np.ndarray:
        return np.zeros(shape[0] if isinstance(shape[0], tuple) else shape, dtype=self.dtype)

    def new_randn(self, *shape: int | Tuple[int, ...]) -> np.ndarray:
        return np.random.randn(*(shape[0] if isinstance(shape[0], tuple) else shape)).astype(self.dtype)

    def concat(self, *arrays: np.ndarray, dim: int) -> np.ndarray:
        return np.concatenate(arrays, axis=dim)

    def normalize(self, arr: SeqArrayLike) -> np.ndarray:
        if self.matches(arr):
            return arr

        if torch.is_tensor(arr):
            arr = arr.detach().cpu().numpy()
        elif not isinstance(arr, np.ndarray):
            arr = np.array(arr, dtype=self.dtype)
        return arr.astype(self.dtype)

    def copy(self, arr: np.ndarray) -> np.ndarray:
        return np.copy(arr)

    def __repr__(self) -> str:
        return f"NumpyArrayInterface(dtype={self.dtype})"


class TensorInterface(ArrayInterface[torch.Tensor]):
    def __init__(self, dtype_like: torch.dtype | torch.Tensor, device: str | torch.device = None):
        if torch.is_tensor(dtype_like):
            self.dtype = dtype_like.dtype
            self.device = dtype_like.device
            if device and torch.device(device) != dtype_like.device:
                raise ValueError(
                    f"Got conflicting device {device} and {dtype_like.device} for the tensor {dtype_like}."
                )
        else:
            self.dtype = dtype_like
            self.device = torch.device(device or "cpu")
            if self.device.index is None and self.device.type == "cuda":
                self.device = torch.device("cuda:0")

    def get(self, arr: torch.Tensor, *idx) -> torch.Tensor:
        return arr.__getitem__(*idx)

    def set(self, arr: torch.Tensor, *idx, value) -> None:
        arr.__setitem__(*idx, value)

    def get_shape(self, arr: torch.Tensor) -> Tuple[int, ...]:
        return tuple(arr.shape)

    def matches(self, arr: SeqArray | SeqDTypeLike) -> bool:
        if super().matches(arr):
            return arr.device == self.device
        return False

    def new_empty(self, *shape: int | Tuple[int, ...]) -> torch.Tensor:
        return torch.empty(*shape, dtype=self.dtype, device=self.device)

    def new_zeros(self, *shape: int | Tuple[int, ...]) -> torch.Tensor:
        return torch.zeros(*shape, dtype=self.dtype, device=self.device)

    def new_randn(self, *shape: int | Tuple[int, ...]) -> torch.Tensor:
        return torch.randn(*shape, dtype=self.dtype, device=self.device)

    def concat(self, *arrays: torch.Tensor, dim: int) -> torch.Tensor:
        return torch.cat(arrays, dim=dim)

    def normalize(self, arr: SeqArrayLike) -> torch.Tensor:
        if self.matches(arr):
            return arr

        if isinstance(arr, np.ndarray):
            arr = torch.from_numpy(arr)
        elif not torch.is_tensor(arr):
            arr = torch.tensor(arr, dtype=self.dtype, device=self.device)
        return arr.to(self.dtype, device=self.device)

    def copy(self, arr: torch.Tensor) -> torch.Tensor:
        return arr.clone()

    def __repr__(self) -> str:
        return f"TensorInterface(dtype={self.dtype}, device={self.device})"
