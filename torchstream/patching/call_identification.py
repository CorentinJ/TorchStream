import sys
from typing import List, Tuple


def get_callstack_locs() -> List[Tuple[str, str, int]]:
    frame = sys._getframe()
    out = []
    while frame is not None:
        out.append((frame.f_code.co_filename, frame.f_code.co_name, frame.f_lineno))
        frame = frame.f_back
    return out


def get_relative_callstack_locs(parent_stack: List[Tuple[str, str, int]]) -> List[Tuple[str, str, int]]:
    child_stack = get_callstack_locs()

    if child_stack[: len(parent_stack)] != parent_stack:
        raise ValueError("The provided parent stack is not a prefix of the current stack.")

    return child_stack[len(parent_stack) :]


def get_fully_qualified_name(obj: str | object) -> str:
    if isinstance(obj, str):
        return obj
    return obj.__module__ + "." + obj.__qualname__
