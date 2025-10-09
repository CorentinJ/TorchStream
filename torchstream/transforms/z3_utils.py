import math
from typing import Tuple, Union

from z3 import ArithRef, If

IntLike = Union[int, ArithRef]


def z3_ceil_div(a: IntLike, b: IntLike) -> IntLike:
    """Ceiling division that works for both Python ints and z3 expressions."""
    if isinstance(a, int) and isinstance(b, int):
        return int(math.ceil(a / b))
    return If(a >= 0, (a + b - 1) / b, -((-a) / b))


def z3_floor_div(a: IntLike, b: IntLike) -> IntLike:
    """Floor division that works for both Python ints and z3 expressions."""
    if isinstance(a, int) and isinstance(b, int):
        return a // b
    return If(a >= 0, a / b, -((-a + b - 1) / b))


def z3_max(a: IntLike, b: IntLike) -> IntLike:
    """max() for both Python ints and z3 expressions."""
    if isinstance(a, int) and isinstance(b, int):
        return max(a, b)
    return If(a > b, a, b)


def z3_min(a: IntLike, b: IntLike) -> IntLike:
    """min() for both Python ints and z3 expressions."""
    if isinstance(a, int) and isinstance(b, int):
        return min(a, b)
    return If(a < b, a, b)


def z3_divmod(a: IntLike, b: IntLike) -> Tuple[IntLike, IntLike]:
    """Divmod that works for both Python ints and z3 expressions."""
    if isinstance(a, int) and isinstance(b, int):
        return divmod(a, b)
    q = z3_floor_div(a, b)
    r = a - q * b
    return q, r
