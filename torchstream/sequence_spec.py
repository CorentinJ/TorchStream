import numbers
from typing import Tuple, Union, overload

import numpy as np
import torch

from torchstream.sequence_dtype import is_similar_dtype, seqdtype

# TODO: include python base numerical types as List
# TODO: limit to numerical types (i.e. not strings)
Sequence = Union[torch.Tensor, np.ndarray]


# TODO: dtypes test for this class
class SeqSpec:
    @overload
    def __init__(
        self,
        shape: Tuple[int, ...],
        dtype: seqdtype = torch.float32,
        device: str = None,
    ) -> None: ...

    @overload
    def __init__(
        self,
        seq_dim: int,
        ndims: int = None,
        dtype: seqdtype = torch.float32,
        device: str = None,
    ) -> None: ...

    def __init__(self, *args, dtype: seqdtype = torch.float32, device: str = None):
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

        self.dtype = dtype

        if self.is_torch and not device:
            self.device = "cpu"
        elif not self.is_torch and device:
            raise ValueError(f"device is only valid for torch tensors, got {device} for a {dtype} sequence")
        self.device = device

    @property
    def is_torch(self) -> bool:
        return isinstance(self.dtype, torch.dtype)

    @property
    def is_numpy(self) -> bool:
        return isinstance(self.dtype, np.dtype)

    # TODO: needs heavy testing
    def matches(self, seq: Sequence) -> Tuple[bool, str]:
        """
        Returns whether a sequence matches the specification. If not, returns a string describing the mismatch.
        """
        if self.is_torch and not torch.is_tensor(seq):
            return False, f"not a tensor (got {type(seq)})"
        elif self.is_numpy and not isinstance(seq, np.ndarray):
            return False, f"not a numpy array (got {type(seq)})"

        if not is_similar_dtype(self.dtype, seq.dtype):
            return False, f"dtype mismatch (got {seq.dtype}, expected {self.dtype})"

        if self.shape:
            if len(seq.shape) != len(self.shape):
                return False, f"shape ndim mismatch (got {seq.shape}, expected {self.shape})"
            for i, (dim_size, expected_dim_size) in enumerate(zip(seq.shape, self.shape)):
                if expected_dim_size is not None and i != self.seq_dim and dim_size != expected_dim_size:
                    return False, f"shape mismatch on dimension {i} (got {seq.shape}, expected {self.shape})"

        if self.device and seq.device != self.device:
            return False, f"device mismatch (got {seq.device}, expected {self.device})"

        return True, ""

    def get_seq_size(self, seq: Sequence) -> int:
        """
        Returns the size of the sequence dimension in the given sequence. If the sequence does not match the
        specification, raises an error.
        """
        matches, msg = self.matches(seq)
        if not matches:
            raise ValueError(f"Failed to get sequence size: {msg}")

        return seq.shape[self.seq_dim]

    def get_shape_for_seq_size(self, seq_size: int) -> Tuple[int, ...]:
        if seq_size < 0:
            raise ValueError(f"Sequence size must be non-negative, got {seq_size}")
        if self.shape is None or any(dim is None for dim in self.shape):
            raise ValueError(
                f"Cannot sample from a sequence specification with unknown dimensions. Shape is {self.shape}"
            )

        shape = list(self.shape)
        shape[self.seq_dim] = seq_size
        return tuple(shape)

    def randn(self, seq_size: int) -> Sequence:
        """
        Sample a sequence of the given size from a normal distribution (discretized for integer types).
        """
        shape = self.get_shape_for_seq_size(seq_size)
        if self.is_torch:
            return torch.randn(shape, dtype=self.dtype, device=self.device)
        else:
            return np.random.randn(*shape).astype(self.dtype)

    def empty(self, seq_size: int = 0) -> Sequence:
        """
        Returns an empty sequence (i.e. uninitialized data) of the given size.
        """
        shape = self.get_shape_for_seq_size(seq_size)
        if self.is_torch:
            return torch.empty(shape, dtype=self.dtype, device=self.device)
        else:
            return np.empty(shape, dtype=self.dtype)

    def __repr__(self) -> str:
        return f"SeqSpec(shape={self.shape}, dtype={self.dtype}, device={self.device})"
