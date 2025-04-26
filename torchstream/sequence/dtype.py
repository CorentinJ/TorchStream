import numpy as np
import torch
from numpy.typing import ArrayLike, DTypeLike

"""
Notes on typing:
- In torch, a "dtype" is a scalar type such as float32, in64, etc... In numpy, a "dtype" is a rich instance containing 
a similar scalar type but also memory layout information, byte encoding order, etc...
"""

# Used for argument typing. Supports a broad range of types.
SeqDTypeLike = torch.dtype | DTypeLike
SeqArrayLike = torch.Tensor | ArrayLike

# Used for normalized types (e.g. class attributes)
seqdtype = torch.dtype | np.dtype


# TODO: clean overload for type vs array
def to_seqdtype(obj: SeqDTypeLike | SeqArrayLike) -> seqdtype:
    """
    Normalizes a dtype to a torch.dtype or a numpy dtype. Can also extract the dtype from an array-like object.
    """
    if isinstance(obj, torch.dtype):
        return obj
    elif torch.is_tensor(obj):
        return obj.dtype
    try:
        return np.dtype(obj)
    except TypeError:
        raise TypeError(
            f"Cannot convert {obj} to a sequence dtype. Please use a torch.dtype or a numpy dtype."
        ) from None


def dtypes_compatible(dtype1: SeqDTypeLike, dtype2: SeqDTypeLike) -> bool:
    """
    Returns whether the two dtypes are compatible for broadcasting or concatenation. They must both be either torch or
    numpy dtypes, and be of the same scalar kind (floating point, integer, ...)
    """
    dtype1, dtype2 = to_seqdtype(dtype1), to_seqdtype(dtype2)

    if isinstance(dtype1, torch.dtype) and isinstance(dtype2, torch.dtype):
        return _is_similar_torch_dtype(dtype1, dtype2)
    elif isinstance(dtype1, np.dtype) and isinstance(dtype2, np.dtype):
        return dtype1.kind == dtype2.kind
    else:
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
