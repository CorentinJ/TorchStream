import numbers
from typing import Sequence, Tuple, overload

import numpy as np
import torch

from torchstream.sequence.array_interface import ArrayInterface
from torchstream.sequence.dtype import DeviceLike, SeqArrayLike, SeqDTypeLike, to_seqdtype


@overload
def get_shape_and_array_interface(
    *shape: int,
    dtype: SeqDTypeLike = torch.float32,
    device: DeviceLike = "cpu",
) -> Tuple[Tuple[int, ...], ArrayInterface]: ...
@overload
def get_shape_and_array_interface(
    shape: Sequence[int],
    dtype: SeqDTypeLike = torch.float32,
    device: DeviceLike = "cpu",
) -> Tuple[Tuple[int, ...], ArrayInterface]: ...
@overload
def get_shape_and_array_interface(
    array: SeqArrayLike,
    seq_dim: int = -1,
) -> Tuple[Tuple[int, ...], ArrayInterface]: ...
@overload
def get_shape_and_array_interface(
    shape: Sequence[int],
    arr_if: ArrayInterface,
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

    # Otherwise we're in the other overloads
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

        if len(spec) == 1 and isinstance(spec[0], ArrayInterface):
            arr_if = spec[0]
        else:
            device = kwargs.pop("device", None)
            dtype = kwargs.pop("dtype", torch.float32)
            if kwargs:
                raise ValueError(f"Unexpected keyword arguments {kwargs} when passing shape dimensions")
            for remaining_arg in spec:
                if isinstance(remaining_arg, (str, torch.device)):
                    device = remaining_arg
                else:
                    dtype = to_seqdtype(remaining_arg)
            arr_if = ArrayInterface(dtype, device)

    # Verify the shape
    if sum(1 for dim in shape if dim <= -1) != 1:
        raise ValueError(f"Shape must have exactly one negative (=sequence) dimension, got {shape}")
    if any(dim == 0 for dim in shape):
        raise ValueError(f"Shape dimensions cannot be 0, got {shape}")

    return shape, arr_if


# TODO: needs heavy testing
def array_matches_shape_and_type(
    arr: SeqArrayLike, seq_shape: Tuple[int, ...], arr_if: ArrayInterface
) -> Tuple[bool, str]:
    """
    Returns whether a given array is compatible with the sequence specification. Compatible in this context means
    that, at least, the array:
        - is from the same library as the specification (torch, numpy, ...)
        - has the same number representation type (floating point, integer, complex, ...) as the sequence dtype
        - matches the shape of the specification (except for the sequence dimension, which is a strictly
        negative integer)
    """
    if not arr_if.matches(arr):
        device_str = f" {arr.device}" if isinstance(arr, torch.Tensor) else ""
        return False, f"library or dtype mismatch, got{device_str} {arr.dtype} for {arr_if}"

    if len(arr.shape) != len(seq_shape):
        return False, f"shape ndim mismatch (got {arr.shape}, expected {seq_shape})"

    seq_dim = next((i for i, dim_size in enumerate(seq_shape) if dim_size <= -1), None)
    for i, (dim_size, expected_dim_size) in enumerate(zip(arr.shape, seq_shape)):
        if i == seq_dim:
            if dim_size % (-expected_dim_size) != 0:
                return (
                    False,
                    f"sequence dimension is not a multiple of expected size (got {dim_size}, expected "
                    f"a shape like {tuple(seq_shape)})",
                )

        elif dim_size != expected_dim_size:
            return (
                False,
                f"shape mismatch on dimension {i}: got {tuple(arr.shape)}, expected a shape like {tuple(seq_shape)}",
            )

    return True, ""


def get_shape_for_seq_size(shape: Tuple[int, ...], seq_size: int) -> Tuple[int, ...]:
    """
    Takes a shape with a sequence dimension (indicated by a strictly negative integer), and returns a shape
    with the sequence dimension replaced by the given sequence size. If the sequence dimension is a value other than
    -1, the absolute value of that integer is used as a multiplier for the sequence size. If there is no sequence
    dimension, the shape is returned as-is.
    """
    seq_dim = next(i for i, dim_size in enumerate(shape) if dim_size == -1)
    shape = list(shape)
    shape[seq_dim] = seq_size * (-shape[seq_dim])
    return tuple(shape)
