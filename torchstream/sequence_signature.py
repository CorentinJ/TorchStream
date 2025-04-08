import numbers
from typing import Tuple, Union, overload

import numpy as np
import torch

# TODO: include python base numerical types as List
Sequence = Union[torch.Tensor, np.ndarray]
seqdtype = Union[torch.dtype, np.dtype]


# FIXME: name
class SeqSig:
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
        # Shape overload
        if not isinstance(args[0], numbers.Number):
            self.shape = tuple(int(dim_size) for dim_size in args[0])
            if not self.shape.count(-1) == 1:
                raise ValueError(f"Shape must have a single -1, got {self.shape}")
            self.seq_dim = self.shape.index(-1)
            self.ndim = len(self.shape)

        # Seqdim overload
        else:
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

    # FIXME: rewrite
    def get_tensor(self, sequence_size: int) -> Sequence:
        shape = list(self.shape)
        shape[self.dim] = sequence_size

        if isinstance(self.dtype, torch.dtype):
            return torch.randn(shape, dtype=self.dtype, device=self.device)
        else:
            return np.random.randn(*shape).astype(self.dtype)
