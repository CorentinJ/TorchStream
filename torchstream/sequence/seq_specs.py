import numbers
from typing import Tuple

import numpy as np
import torch
from numpy.typing import ArrayLike, DTypeLike

from torchstream.sequence.dtype import seqdtype


class SeqSpec:
    dtype: seqdtype

    # TODO! overloads in pyi
    def __new__(cls, *args, dtype: torch.dtype | DTypeLike = torch.float32, device: str = None):
        if cls is SeqSpec:
            if isinstance(dtype, torch.dtype):
                cls = TensorSeqSpec
            try:
                np.dtype(dtype)
                cls = ArraySeqSpec
            except TypeError:
                raise TypeError(
                    f"No sequence implementation exists for dtype {dtype}. Please provide a correct dtype."
                ) from None

        return object.__new__(cls)

    # TODO! overloads in pyi
    def __init__(self, *args):
        """
        TODO: doc
        """
        # Shape overload
        if not isinstance(args[0], numbers.Number):
            self.shape = tuple(int(dim_size) for dim_size in args[0])
            if not self.shape.count(-1) == 1:
                raise ValueError(f"Shape must have a single -1, got {self.shape}")
            self.seq_dim = self.shape.index(-1)
            self.ndim = len(self.shape)

        # Seqdim overload
        else:
            # TODO: handle negative dims?
            self.seq_dim = args[0]
            self.ndim = args[1] if len(args) > 1 else None
            if self.ndim:
                if self.seq_dim >= self.ndim:
                    raise ValueError(f"seq_dim {self.seq_dim} must be less than ndims {self.ndim}")
                self.shape = (None,) * self.ndim
                self.shape[self.seq_dim] = -1
            else:
                self.shape = None

    # TODO: needs heavy testing
    def is_compatible(self, arr: ArrayLike) -> Tuple[bool, str]:
        """
        Returns whether a given array is compatible with the sequence specification. Compatible in this context means
        that, at least, the array:
            - is from the same library as the specification (torch, numpy, ...)
            - has the same number representation type (floating point, integer, complex, ...) as the sequence dtype
            - matches the shape of the specification (except for the sequence dimension, which is -1), or the number of
            dimensions when the shape is not specified.

        If the array is not compatible, returns a string describing the mismatch.
        """
        raise NotImplementedError()


# TODO: limit to numerical types (i.e. not strings)
#   -> Why though? For the NaN trick?


class ArraySeqSpec(SeqSpec):
    def __init__(self, *args, dtype: DTypeLike = np.float64):
        """
        TODO: doc
        """
        self.dtype = np.dtype(dtype)

    # TODO: needs heavy testing
    def matches(self, seq: Sequence) -> Tuple[bool, str]:
        """
        Returns whether a sequence matches the specification. If not, returns a string describing the mismatch.
        """
        # TODO!!
        raise NotImplementedError()
        # if self.is_torch and not torch.is_tensor(seq):
        #     return False, f"not a tensor (got {type(seq)})"
        # elif self.is_numpy and not isinstance(seq, np.ndarray):
        #     return False, f"not a numpy array (got {type(seq)})"

        # if not dtypes_compatible(self.dtype, seq.dtype):
        #     return False, f"dtype mismatch (got {seq.dtype}, expected {self.dtype})"

        # if self.shape:
        #     if len(seq.shape) != len(self.shape):
        #         return False, f"shape ndim mismatch (got {seq.shape}, expected {self.shape})"
        #     for i, (dim_size, expected_dim_size) in enumerate(zip(seq.shape, self.shape)):
        #         if expected_dim_size is not None and i != self.seq_dim and dim_size != expected_dim_size:
        #             return False, f"shape mismatch on dimension {i} (got {seq.shape}, expected {self.shape})"

        # if self.device and seq.device != self.device:
        #     return False, f"device mismatch (got {seq.device}, expected {self.device})"

        # return True, ""
