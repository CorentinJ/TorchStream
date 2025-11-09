import sys
from types import CodeType, FrameType
from typing import Callable, Iterable, Optional, Tuple, Union

import torch

# What we keep for each frame; adjust as needed
FrameKey = Tuple[str, str, int, int]  # (filename, funcname, lineno, co_firstlineno)

Predicate = Callable[[FrameType], bool]
Ref = Union[Predicate, CodeType, Callable, str, None]


def _make_predicate(ref: Ref) -> Predicate:
    """
    ref may be:
      - code object -> stop when frame.f_code is ref
      - function/method -> stop when frame.f_code is func.__code__
      - module name (str) -> stop when frame's __name__ equals ref
      - filename suffix (str) -> stop when filename endswith(ref)
      - predicate(frame) -> used directly
      - None -> never stop (only max_depth bounds the walk)
    """
    if ref is None:
        return lambda f: False

    if callable(ref) and hasattr(ref, "__code__"):
        code = ref.__code__  # function/method
        return lambda f, code=code: f.f_code is code

    if isinstance(ref, CodeType):
        return lambda f, code=ref: f.f_code is code

    if isinstance(ref, str):
        # Treat as module name OR filename suffix
        def _pred(f, s=ref):
            mod = f.f_globals.get("__name__")
            return (mod == s) or f.f_code.co_filename.endswith(s)

        return _pred

    # Already a predicate
    if callable(ref):
        return ref  # type: ignore[arg-type]

    raise TypeError(f"Unsupported ref type: {type(ref)!r}")


def stack_slice_until(
    ref: Ref, include_ref: bool = False, max_depth: int = 64, require_ref: bool = True
) -> Tuple[FrameKey, ...]:
    """
    Walk back from caller, building a key until `ref` matches.
    If `require_ref` and the sentinel isn't found, raise RuntimeError.
    """
    pred = _make_predicate(ref)
    f = sys._getframe(1)
    out: list[FrameKey] = []
    hit = False

    for _ in range(max_depth):
        if f is None:
            break
        if pred(f):
            hit = True
            if include_ref:
                out.append((f.f_code.co_filename, f.f_code.co_name, f.f_lineno, f.f_code.co_firstlineno))
            break
        out.append((f.f_code.co_filename, f.f_code.co_name, f.f_lineno, f.f_code.co_firstlineno))
        f = f.f_back

    if require_ref and not hit:
        raise RuntimeError("Reference frame not found in current call stack")

    return tuple(out)


from contextlib import AbstractContextManager
from unittest.mock import patch


class SelectivePatch(AbstractContextManager):
    def __init__(self, target: str, dispatch: dict):
        """
        target: 'pkg.mod.func' to patch.
        dispatch: mapping from stack keys to replacement callables.
                  use dispatch.get(None) as a default.
        """
        self._target = target
        self._dispatch = dispatch
        self._patcher = None
        self._sentinel_code: Optional[CodeType] = None

    def __enter__(self):
        # Sentinel = function where the with-statement lives
        caller = sys._getframe(1)
        self._sentinel_code = caller.f_code

        self._patcher = patch(self._target, new=self._wrapper)
        self._patcher.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._patcher:
            self._patcher.stop()
            self._patcher = None
        self._sentinel_code = None
        return False

    def _wrapper(self, *args, **kwargs):
        # Build a key from the call site up to the with-site; require the sentinel
        key = stack_slice_until(self._sentinel_code, include_ref=False, max_depth=48, require_ref=True)
        fn = self._dispatch.get(key, self._dispatch.get(None))
        if fn is None:
            # Fall back to original if exposed by your patcher,
            # or raise to surface missing routing.
            raise RuntimeError("No replacement found for this call site")
        return fn(*args, **kwargs)


def work_fn():
    a = torch.arange(5)
    b = torch.cumsum(a, dim=0)
    c = b // 3
    d = torch.cumsum(c, dim=0)
    return d - a


if __name__ == "__main__":

    def handler(orig_fn, args, kwargs):
        print("Intercepted call to:", orig_fn)
        return orig_fn(*args, **kwargs)

    with SelectivePatch(target="torch.cumsum", dispatch={None: handler}):
        result = work_fn()
        print("Result:", result)
