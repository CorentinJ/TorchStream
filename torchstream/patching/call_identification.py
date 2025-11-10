import sys
from typing import List, Tuple


def get_callstack_locs(skip_n_first=0) -> List[Tuple[str, str, int]]:
    frame = sys._getframe(skip_n_first + 1)
    out = []
    while frame is not None:
        out.append((frame.f_code.co_filename, frame.f_code.co_name, frame.f_lineno))
        frame = frame.f_back
    return out[::-1]


def get_relative_callstack_locs(parent_stack: List[Tuple[str, str, int]], skip_n_first=0) -> List[Tuple[str, str, int]]:
    child_stack = get_callstack_locs(skip_n_first=skip_n_first + 1)

    parent_stack = list(parent_stack)
    while parent_stack and child_stack[0] == parent_stack[0]:
        child_stack = child_stack[1:]
        parent_stack = parent_stack[1:]

    return child_stack
