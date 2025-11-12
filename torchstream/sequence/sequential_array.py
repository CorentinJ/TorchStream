import numbers
from typing import Sequence as _Sequence
from typing import Tuple, overload

import numpy as np
import torch

from torchstream.sequence.array_interface import ArrayInterface
from torchstream.sequence.dtype import DeviceLike, SeqArrayLike, SeqDTypeLike, to_seqdtype


@overload
def get_shape_and_array_interface(
    *shape: int, dtype: SeqDTypeLike = torch.float32, device: DeviceLike = "cpu"
) -> Tuple[Tuple[int, ...], ArrayInterface]: ...
@overload
def get_shape_and_array_interface(
    shape: _Sequence[int], /, dtype: SeqDTypeLike = torch.float32, device: DeviceLike = "cpu"
) -> Tuple[Tuple[int, ...], ArrayInterface]: ...
@overload
def get_shape_and_array_interface(
    array: SeqArrayLike, /, seq_dim: int = -1
) -> Tuple[Tuple[int, ...], ArrayInterface]: ...
def get_shape_and_array_interface(*spec, **kwargs) -> Tuple[Tuple[int, ...], ArrayInterface]:
    """
    TODO: doc
    """
    # First arg is an array, that's the third overload
    if torch.is_tensor(spec[0]) or isinstance(spec[0], np.ndarray):
        if len(spec) == 1:
            seq_dim = kwargs.pop("seq_dim", -1)
        elif len(spec) == 2:
            seq_dim = spec[1]
        else:
            raise ValueError(f"Expected an array and optional seq_dim for argument, got {spec}")
        if kwargs:
            raise ValueError(f"Unexpected keyword arguments {kwargs} when passing an array")

        arr_if = ArrayInterface(spec[0])
        shape = list(arr_if.get_shape(spec[0]))
        shape[seq_dim] = -1
        shape = tuple(shape)

    # Otherwise we're in the first two overloads
    else:
        if not isinstance(spec[0], numbers.Number):
            if not isinstance(spec[0], (list, tuple)):
                raise ValueError(f"Shape must be a sequence of integers, got {spec[0]}")
            shape = tuple(spec[0])
            spec = spec[1:]
        else:
            split_idx = next((i for i, s in enumerate(spec) if not isinstance(s, numbers.Number)), len(spec))
            shape = tuple(int(dim_size) for dim_size in spec[:split_idx])
            spec = spec[split_idx:]

        device = kwargs.pop("device", "cpu")
        dtype = kwargs.pop("dtype", torch.float32)
        for remaining_arg in spec:
            if isinstance(remaining_arg, (str, torch.device)):
                device = remaining_arg
            else:
                dtype = to_seqdtype(remaining_arg)

        arr_if = ArrayInterface(dtype, device)

    # Verify the shape
    if sum(1 for dim in shape if dim <= -1) > 1:
        raise ValueError(f"Shape must have at most one negative (=sequence) dimension, got {shape}")
    if any(dim == 0 for dim in shape):
        raise ValueError(f"Shape dimensions cannot be 0, got {shape}")

    return shape, arr_if


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


def get_shape_for_size(self, seq_size: int) -> Tuple[int, ...]:
    shape = list(self.shape)
    shape[self.seq_dim] = seq_size
    return tuple(shape)
