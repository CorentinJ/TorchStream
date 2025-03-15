from collections import abc
from typing import Union

import numpy as np
import torch

TensorLike = Union[torch.Tensor, np.ndarray]


class TensorProvider(abc.ABC):
    def __init__(self, dim: int):
        self.dim = dim

    @abc.abstractmethod
    def get_tensor(self, sequence_size: int) -> TensorLike:
        pass


class TensorSpec(TensorProvider):
    def __init__(
        self,
        shape: tuple,
        dtype: Union[torch.dtype, np.dtype] = torch.float32,
        device: str = None,
    ):
        # TODO: checks
        self.shape = shape
        self.dtype = dtype
        self.device = device

        super().__init__(dim=list(shape).index(-1))

    def get_tensor(self, sequence_size: int) -> TensorLike:
        shape = list(self.shape)
        shape[self.dim] = sequence_size

        if isinstance(self.dtype, torch.dtype):
            return torch.randn(shape, dtype=self.dtype, device=self.device)
        else:
            return np.random.randn(*shape).astype(self.dtype)
