import numbers
from typing import Sequence, Tuple, TypeAlias, overload

import torch

from torchstream.sequence.array_interface import ArrayInterface, SeqArray
from torchstream.sequence.dtype import SeqArrayLike, SeqDTypeLike

class SeqSpec:
    @overload
    def __init__(self, *shape: int) -> None: ...

    @overload
    def __init__(self, *shape: ShapeArg, dtype: SeqDTypeLike) -> None: ...

    @overload
    def __init__(self, *shape: ShapeArg, dtype: SeqDTypeLike, device: DeviceLike) -> None: ...

    @overload
    def __init__(self, array: SeqArrayLike, seq_dim: int, /) -> None: ...

    @overload
    def __init__(self, *specs: SpecTuple) -> None: ...
    def __init__(
        self,
        *specs,
        dtype: SeqDTypeLike = torch.float32,
        device: OptionalDevice = None,
    ):
        """
        TODO: doc
        """
        if not isinstance(shape[0], numbers.Number):
            if not isinstance(shape[0], (list, tuple)):
                raise ValueError(f"Shape must be a list or tuple of integers, got {shape[0]}")
            shape = shape[0]

        self.shape = tuple(int(dim_size) for dim_size in shape)
        if not self.shape.count(-1) == 1:
            raise ValueError(f"Shape must have a single -1, got {self.shape}")
        self.seq_dim = self.shape.index(-1)
        self.ndim = len(self.shape)

        self._arr_if = ArrayInterface(dtype, device)

    # TODO: needs heavy testing
    def matches(self, arr: SeqArrayLike) -> Tuple[bool, str]:
        """
        Returns whether a given array is compatible with the sequence specification. Compatible in this context means
        that, at least, the array:
            - is from the same library as the specification (torch, numpy, ...)
            - has the same number representation type (floating point, integer, complex, ...) as the sequence dtype
            - matches the shape of the specification (except for the sequence dimension, which is -1), or the number of
            dimensions when the shape is not specified.
        """
        if not self._arr_if.matches(arr):
            device_str = f" {arr.device}" if isinstance(arr, torch.Tensor) else ""
            return False, f"library or dtype mismatch, got{device_str} {arr.dtype} for {self._arr_if}"

        if self.shape:
            if len(arr.shape) != len(self.shape):
                return False, f"shape ndim mismatch (got {arr.shape}, expected {self.shape})"
            for i, (dim_size, expected_dim_size) in enumerate(zip(arr.shape, self.shape)):
                if expected_dim_size is not None and i != self.seq_dim and dim_size != expected_dim_size:
                    return (
                        False,
                        f"shape mismatch on dimension {i}: got {tuple(arr.shape)}, "
                        f"expected a shape like {tuple(self.shape)}",
                    )
        else:
            pass  # TODO!

        return True, ""

    def get_shape_for_size(self, size: int) -> Tuple[int, ...]:
        shape = list(self.shape)
        shape[self.seq_dim] = size
        return tuple(shape)

    def new_empty(self, seq_size: int = 0) -> SeqArray:
        """
        Returns an empty array of the given shape. The array's values are uninitialized.
        """
        return self._arr_if.new_empty(self.get_shape_for_size(seq_size))

    def new_zeros(self, seq_size: int) -> SeqArray:
        """
        Returns an array of the given size, filled with zeros.
        """
        return self._arr_if.new_zeros(self.get_shape_for_size(seq_size))

    def new_randn(self, seq_size: int) -> SeqArray:
        """
        Sample an array of the given size from a normal distribution (discretized for integer types).
        """
        return self._arr_if.new_randn(self.get_shape_for_size(seq_size))
