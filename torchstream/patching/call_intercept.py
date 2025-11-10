import importlib
from copy import deepcopy
from typing import Callable

from torchstream.patching.call_identification import (
    get_callstack_locs,
    get_relative_callstack_locs,
)


def get_fully_qualified_name(obj: str | object) -> str:
    if isinstance(obj, str):
        return obj
    return obj.__module__ + "." + obj.__qualname__


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
        handler_fn: Callable | None = None,
        store_in_out: bool = False,
        deep_copy_in_out: bool = True,
        pass_original_fn: bool = False,
        pass_callstack_locs: bool = False,
    ):
        """
        Context manager to intercept calls to a target function and redirect them to a handler function.

        :param target_fn: The target function to intercept. Prefer to provide this function as a string of its
        fully qualified name (e.g., "module.submodule.function" or "module.submodule.Class.method"). Providing the
        function object directly is supported but may fail to resolve.
        :param handler_fn: The handler function that will be called instead of the target function. It will be given
        the same arguments as the original function and its return value will be returned in place of the original
        function's return value. If None, the original function will be called instead, which is useful to
        store call inputs and outputs without modifying behavior.
        :param store_in_out: If True, the context manager will store the inputs and outputs of each intercepted call.
        :param deep_copy_in_out: If True, the stored inputs and outputs will be deep-copied before being stored.
        :param pass_original_fn: If True, the original function will be passed to the handler function as a keyword
        argument named 'original_fn'.
        :param pass_callstack_locs: If True, a list of triplets (filename: str, function_name: str, line_number: int)
        will be passed as a keyword argument named 'callstack_locs' to the handler function. This list represents
        call stack frame between where this context manager's __enter__ was called and the intercepted function call.
        It serves as a way to identify the call site relative to the context manager.
        """
        self._target_fqn = get_fully_qualified_name(target_fn)
        self._handler_fn = handler_fn
        self._pass_callstack_locs = pass_callstack_locs
        self._pass_original_fn = pass_original_fn
        self._store_in_out = store_in_out
        self._deep_copy_in_out = deep_copy_in_out

        self._call_in_outs = []
        self._callstack_reference = None
        self._target_owner = None
        self._target_attr_name = None
        self._original_fn = None

    @property
    def call_in_outs(self) -> list[tuple[tuple, dict, object]]:
        """
        Returns the stored call inputs and outputs as a list of tuples (in_args, in_kwargs, out).
        """
        if not self._store_in_out:
            raise RuntimeError(
                "Call in-out storage was not enabled. "
                "Set store_in_out=True when creating the intercept_calls context manager to enable it."
            )
        return self._call_in_outs

    def __enter__(self):
        # Mark where we are being called from in the callstack
        self._callstack_reference = get_callstack_locs(skip_n_first=1)

        # Obtain the original function
        self._target_owner, self._target_attr_name = retrieve_object(self._target_fqn)

        def wrapper(*args, **kwargs):
            if self._pass_callstack_locs:
                kwargs["callstack_locs"] = get_relative_callstack_locs(self._callstack_reference, skip_n_first=1)
            if self._pass_original_fn:
                kwargs["original_fn"] = self._original_fn

            out = (self._handler_fn or self._original_fn)(*args, **kwargs)

            if self._store_in_out:
                stored_call = (args, kwargs, out)
                if self._deep_copy_in_out:
                    stored_call = deepcopy(stored_call)
                self._call_in_outs.append(stored_call)

            return out

        # Patch it
        self._original_fn = getattr(self._target_owner, self._target_attr_name)
        setattr(self._target_owner, self._target_attr_name, wrapper)

        return self

    def __exit__(self, exc_type, exc, tb):
        # Undo the patch
        setattr(self._target_owner, self._target_attr_name, self._original_fn)

        return False
