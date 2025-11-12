from typing import Sequence as _Sequence
from typing import Tuple, overload

import numpy as np
import torch

from torchstream.sequence.array_interface import SeqArray, TensorInterface
from torchstream.sequence.dtype import DeviceLike, SeqArrayLike, SeqDTypeLike
from torchstream.sequence.sequential_array import (
    array_matches_shape_and_type,
    get_shape_and_array_interface,
    get_shape_for_seq_size,
)


class SeqSpec:
    @overload
    def __init__(self, *shape: int, dtype: SeqDTypeLike = torch.float32, device: DeviceLike = "cpu") -> None: ...
    @overload
    def __init__(
        self, shape: _Sequence[int], dtype: SeqDTypeLike = torch.float32, device: DeviceLike = "cpu"
    ) -> None: ...
    @overload
    def __init__(self, array: SeqArrayLike, seq_dim: int = -1) -> None: ...
    @overload
    def __init__(self, *specs: Tuple) -> None: ...
    def __init__(self, *specs, **kwargs) -> None:
        """
        TODO: doc
        """
        if all(isinstance(spec, tuple) for spec in specs) and not kwargs:
            self.specs = [get_shape_and_array_interface(*spec) for spec in specs]
        else:
            self.specs = [get_shape_and_array_interface(*specs, **kwargs)]

    def matches(self, *arrs: SeqArrayLike) -> Tuple[bool, str]:
        """
        Returns whether the given arrays are compatible with the sequence specification. Compatible in this context
        means that, at least, all arrays:
            - are each from the same library as the specification (torch, numpy, ...)
            - have the same number representation type (floating point, integer, complex, ...) as their respective
            sequence dtype
            - match the shape of the specification (except for the sequence dimension which is a strictly
            negative integer)

        The arrays must be provided in the same order as the specification.
        """
        if len(arrs) != len(self.specs):
            try:
                shape_str = " with shapes [" + ", ".join(str(tuple(a.shape)) for a in arrs) + "]"
            except Exception:
                shape_str = ""
            return False, f"specification {self} expects {len(self.specs)} arrays, got {len(arrs)}{shape_str}"

        for idx, (arr, (shape, arr_if)) in enumerate(zip(arrs, self.specs)):
            matches, reason = array_matches_shape_and_type(arr, shape, arr_if)
            if not matches:
                return False, f"array #{idx}: {reason}" if len(self.specs) > 1 else reason

        return True, ""

    def get_shapes_for_seq_size(self, seq_size: int) -> Tuple[Tuple[int, ...], ...]:
        """
        Returns the shapes of all arrays in the specification for a sequence of the given size. Each shape is returned
        with the sequence dimension replaced by the given sequence size. If the sequence dimension is a value other than
        -1, the absolute value of that integer is used as a multiplier for the sequence size. If there is no sequence
        dimension, the shape is returned as-is.

        Example
        --------
        >>> spec = SeqSpec((-1, 10, 15), (8, -2))
        >>> spec.get_shapes_for_seq_size(14)
        ((14, 10, 15), (8, 28))
        """
        return tuple([get_shape_for_seq_size(shape, seq_size) for shape, _ in self.specs])

    def new_empty(self, seq_size: int = 0) -> Tuple[SeqArray, ...]:
        """
        Returns empty arrays with the given specification. The array's values are uninitialized.
        """
        shapes = self.get_shapes_for_seq_size(seq_size)
        return tuple(arr_if.new_empty(*shape) for shape, (_, arr_if) in zip(shapes, self.specs))

    def new_zeros(self, seq_size: int) -> Tuple[SeqArray, ...]:
        """
        Returns arrays of the given sequence size with the given specification, filled with zeros.
        """
        shapes = self.get_shapes_for_seq_size(seq_size)
        return tuple(arr_if.new_zeros(*shape) for shape, (_, arr_if) in zip(shapes, self.specs))

    def new_randn(self, seq_size: int) -> Tuple[SeqArray, ...]:
        """
        Sample arrays of the given sequence size from a normal distribution (discretized for integer types).
        """
        shapes = self.get_shapes_for_seq_size(seq_size)
        return tuple(arr_if.new_randn(*shape) for shape, (_, arr_if) in zip(shapes, self.specs))

    def __repr__(self) -> str:
        out = ""
        for idx, (shape, arr_if) in enumerate(self.specs):
            device_str = f"{arr_if.device} " if hasattr(arr_if, "device") else ""
            dtype_str = str(arr_if.dtype) if not isinstance(arr_if.dtype, np.dtype) else f"np.{arr_if.dtype}"
            array_str = "Tensor" if isinstance(arr_if, TensorInterface) else " Array"
            out += f"\n   {array_str} #{idx}: {shape} {device_str}{dtype_str}"
        return f"SeqSpec({out}\n)"
