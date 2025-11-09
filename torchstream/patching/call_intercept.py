import importlib
from typing import Callable


class intercept_calls:
    # TODO:
    #   - arg to intercept only once
    #   - boolean for storing the call
    #   - return value for the call?
    def __init__(self, target: str, handler: Callable):
        self._target = target
        self._handler = handler

        self._owner = None
        self._attr_name = None
        self._original = None

    def __enter__(self):
        parts = self._target.split(".")

        module = None
        remainder = None
        for i in range(len(parts), 0, -1):
            mod_name = ".".join(parts[:i])
            try:
                module = importlib.import_module(mod_name)
            except ModuleNotFoundError:
                continue
            else:
                remainder = parts[i:]
                break

        if module is None or remainder is None:
            raise ImportError(f"Cannot resolve {self._target!r}")

        obj = module
        for name in remainder[:-1]:
            obj = getattr(obj, name)

        attr_name = remainder[-1]
        original = getattr(obj, attr_name)

        self._owner = obj
        self._attr_name = attr_name
        self._original = original

        orig_fn = original

        def wrapper(*args, **kwargs):
            return self._handler(orig_fn, args, kwargs)

        setattr(obj, attr_name, wrapper)
        return self

    def __exit__(self, exc_type, exc, tb):
        setattr(self._owner, self._attr_name, self._original)
        return False
