import importlib
from typing import Callable

from torchstream.patching.call_identification import (
    get_callstack_locs,
    get_fully_qualified_name,
    get_relative_callstack_locs,
)


def retrieve_object(target: str | object):
    target = get_fully_qualified_name(target)
    target_parts = target.split(".")

    # Import the correct module
    module = None
    remainder = None
    for i in range(len(target_parts), 0, -1):
        mod_name = ".".join(target_parts[:i])
        try:
            module = importlib.import_module(mod_name)
        except ModuleNotFoundError:
            continue
        else:
            remainder = target_parts[i:]
            break
    if module is None or remainder is None:
        raise ImportError(f"Cannot resolve {target!r}")

    owner = module
    # If the object's owner is not the module itself, traverse the remaining parts
    for name in remainder[:-1]:
        owner = getattr(owner, name)

    # Return the object owner and the attribute name, let the caller decide if they want to setattr or getattr
    return owner, remainder[-1]


class intercept_calls:
    def __init__(
        self,
        target_fn: str | object,
        handler_fn: Callable,
        pass_original_fn: bool = False,
        pass_callstack_locs: bool = False,
    ):
        self._target_fqn = get_fully_qualified_name(target_fn)
        self._handler_fn = handler_fn
        self._pass_callstack_locs = pass_callstack_locs
        self._pass_original_fn = pass_original_fn

        self._callstack_reference = None
        self._target_owner = None
        self._target_attr_name = None
        self._original_fn = None

    def __enter__(self):
        # Mark where we are in the callstack
        self._callstack_reference = get_callstack_locs()

        # Obtain the original function
        self._target_owner, self._target_attr_name = retrieve_object(self._target_fqn)

        def wrapper(*args, **kwargs):
            if self._pass_callstack_locs:
                kwargs["callstack_locs"] = get_relative_callstack_locs(self._callstack_reference)
            if self._pass_original_fn:
                kwargs["original_fn"] = self._original_fn
            return self._handler_fn(*args, **kwargs)

        # Patch it
        setattr(self._target_owner, self._target_attr_name, wrapper)

        return self

    def __exit__(self, exc_type, exc, tb):
        # Undo the patch
        setattr(self._target_owner, self._target_attr_name, self._original_fn)

        return False
