from functools import lru_cache
from typing import Union

import numpy as np
import torch

seqdtype = Union[torch.dtype, np.dtype]


# FIXME: name, "similar" is vague
@lru_cache
def is_similar_dtype(dtype1: seqdtype, dtype2: seqdtype) -> bool:
    if isinstance(dtype1, torch.dtype) and isinstance(dtype2, torch.dtype):
        return _is_similar_torch_dtype(dtype1, dtype2)
    if isinstance(dtype1, np.dtype) and isinstance(dtype2, np.dtype):
        return _is_similar_numpy_dtype(dtype1, dtype2)
    return False


def _is_similar_torch_dtype(dtype1: torch.dtype, dtype2: torch.dtype) -> bool:
    t1 = torch.empty((), dtype=dtype1)
    t2 = torch.empty((), dtype=dtype2)

    if torch.is_floating_point(t1) and torch.is_floating_point(t2):
        return True

    if torch.is_complex(t1) and torch.is_complex(t2):
        return True

    # NOTE: int and bool will return true with this
    if (
        not torch.is_floating_point(t1)
        and not torch.is_complex(t1)
        and not torch.is_floating_point(t2)
        and not torch.is_complex(t2)
    ):
        return True

    return False


def _is_similar_numpy_dtype(dtype1: np.dtype, dtype2: np.dtype) -> bool:
    return dtype1.kind == dtype2.kind
