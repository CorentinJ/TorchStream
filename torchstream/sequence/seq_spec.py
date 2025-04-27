import numbers

import torch

from torchstream.sequence.array_interface import ArrayInterface
from torchstream.sequence.dtype import SeqArrayLike, SeqDTypeLike


class SeqSpec:
    # TODO! overloads in pyi
    def __init__(self, *shape_args, dtype_like: SeqDTypeLike | SeqArrayLike, device: str | torch.device = None):
        """
        TODO: doc
        """
        # Shape overload
        if not isinstance(shape_args[0], numbers.Number):
            self.shape = tuple(int(dim_size) for dim_size in shape_args[0])
            if not self.shape.count(-1) == 1:
                raise ValueError(f"Shape must have a single -1, got {self.shape}")
            self.seq_dim = self.shape.index(-1)
            self.ndim = len(self.shape)

        # Seqdim overload
        else:
            # TODO: handle negative dims?
            self.seq_dim = shape_args[0]
            self.ndim = shape_args[1] if len(shape_args) > 1 else None
            if self.ndim:
                if self.seq_dim >= self.ndim:
                    raise ValueError(f"seq_dim {self.seq_dim} must be less than ndims {self.ndim}")
                self.shape = (None,) * self.ndim
                self.shape[self.seq_dim] = -1
            else:
                self.shape = None

        self._arr_if = ArrayInterface(dtype_like, device)

    # TODO: needs heavy testing
    def matches(self, arr: SeqArrayLike) -> bool:
        """
        Returns whether a given array is compatible with the sequence specification. Compatible in this context means
        that, at least, the array:
            - is from the same library as the specification (torch, numpy, ...)
            - has the same number representation type (floating point, integer, complex, ...) as the sequence dtype
            - matches the shape of the specification (except for the sequence dimension, which is -1), or the number of
            dimensions when the shape is not specified.
        """
        if not self._arr_if.matches(arr):
            return False

        # f"dtype mismatch (got {arr.dtype}, expected {self.dtype})"

        if self.shape:
            if len(arr.shape) != len(self.shape):
                return False  # , f"shape ndim mismatch (got {arr.shape}, expected {self.shape})"
            for i, (dim_size, expected_dim_size) in enumerate(zip(arr.shape, self.shape)):
                if expected_dim_size is not None and i != self.arr_dim and dim_size != expected_dim_size:
                    return False  # , f"shape mismatch on dimension {i} (got {arr.shape}, expected {self.shape})"
        else:
            pass  # TODO!!

        return True
