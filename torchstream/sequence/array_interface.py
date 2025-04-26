from abc import ABC
from typing import Generic, Tuple, TypeVar

import numpy as np
import torch
from numpy.typing import ArrayLike, DTypeLike

from torchstream.sequence.dtype import SeqArrayLike, SeqDTypeLike, dtypes_compatible, seqdtype, to_seqdtype

SeqArray = TypeVar("SeqArray", torch.Tensor, np.ndarray)


def _to_slice(*idx):
    if isinstance(idx[0], slice):
        return idx[0]
    return slice(*idx)


class ArrayInterface(ABC, Generic[SeqArray]):
    dtype: seqdtype

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

    def get_along_dim(self, array: SeqArray, *idx, dim: int) -> SeqArray:
        """
        Convenience method to index along a single dimension, returning the full space across all other dimensions.
        """
        slices = [slice(None)] * len(self.get_shape(array))
        slices[dim] = _to_slice(*idx)
        return self.get(array, tuple(slices))

    def set(self, arr: SeqArray, *idx, value) -> None:
        raise NotImplementedError()

    def set_along_dim(self, array: SeqArray, *idx, dim: int, value) -> None:
        """
        Convenience method to set values across a slice of a given dimension, including the full space across all other
        dimensions.
        """
        slices = [slice(None)] * len(self.get_shape(array))
        slices[dim] = _to_slice(*idx)
        self.set(array, tuple(slices), value)

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

    def new_randn(self, *shape: int | Tuple[int, ...]) -> SeqArray:
        """
        Sample a sequence of the given size from a normal distribution (discretized for integer types).
        """
        raise NotImplementedError()


class NumpyArrayInterface(ArrayInterface[np.ndarray]):
    def __init__(self, dtype_like: DTypeLike | ArrayLike):
        self.dtype = np.dtype(dtype_like)

    def get(self, arr: np.ndarray, *idx) -> np.ndarray:
        return arr[idx]

    def set(self, arr: np.ndarray, *idx, value) -> None:
        arr[idx] = value

    def get_shape(self, arr: np.ndarray) -> Tuple[int, ...]:
        return arr.shape

    def new_empty(self, *shape: int | Tuple[int, ...]) -> np.ndarray:
        return np.empty(shape, dtype=self.dtype)

    def new_randn(self, *shape: int | Tuple[int, ...]) -> np.ndarray:
        return np.random.randn(*shape).astype(self.dtype)

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


class TensorInterface(ArrayInterface[torch.Tensor]):
    def __init__(self, dtype_like: torch.dtype | torch.Tensor, device: str | torch.device = None):
        if torch.is_tensor(dtype_like):
            self.dtype = dtype_like.dtype
            self.device = dtype_like.device
            if torch.device(device) != dtype_like.device:
                raise ValueError(
                    f"Got conflicting device {device} and {dtype_like.device} for the tensor {dtype_like}."
                )
        else:
            self.dtype = dtype_like
            self.device = torch.device(device or "cpu")

    def get(self, arr: torch.Tensor, *idx) -> torch.Tensor:
        return arr[idx]

    def set(self, arr: torch.Tensor, *idx, value) -> None:
        arr[idx] = value

    def get_shape(self, arr: torch.Tensor) -> Tuple[int, ...]:
        return tuple(arr.shape)

    def new_empty(self, *shape: int | Tuple[int, ...]) -> torch.Tensor:
        return torch.empty(shape, dtype=self.dtype, device=self.device)

    def new_randn(self, *shape: int | Tuple[int, ...]) -> torch.Tensor:
        return torch.randn(shape, dtype=self.dtype, device=self.device)

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
