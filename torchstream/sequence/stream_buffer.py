import logging
from typing import Callable, Iterable, Optional, Sequence, Tuple, overload

import numpy as np
import torch
from opentelemetry import trace

from torchstream.exception_signature import DEFAULT_ZERO_SIZE_EXCEPTIONS, ExceptionWithSubstring, matches_any_exception
from torchstream.sequence.array_interface import ArrayInterface
from torchstream.sequence.dtype import DeviceLike, SeqArrayLike, SeqDTypeLike
from torchstream.sequence.seq_spec import SeqSpec
from torchstream.sequence.sequential_array import get_shape_and_array_interface, get_shape_for_seq_size

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)


class StreamBuffer:
    """
    FIXME! rewrite this doc
    Tensor-like class for buffering multidimensional sequential data. Supports both torch tensors and numpy arrays.

    >>> buff = StreamBuffer(spec=SeqSpec(seq_dim=1))
    >>> buff.feed(torch.arange(6).reshape((3, 2)))
    >>> buff.feed(torch.arange(9).reshape((3, 3)))
    >>> buff.read(4)
    tensor([[0, 1, 0, 1],
            [2, 3, 3, 4],
            [4, 5, 6, 7]])

    This class simplifies a great deal of sliding window operations.

    TODO: transform asserts into exceptions
    """

    @overload
    def __init__(self, *shape: int, dtype: SeqDTypeLike = torch.float32, device: DeviceLike = "cpu") -> None: ...
    @overload
    def __init__(
        self, shape: Sequence[int], dtype: SeqDTypeLike = torch.float32, device: DeviceLike = "cpu"
    ) -> None: ...
    @overload
    def __init__(self, array: SeqArrayLike, seq_dim: int = -1) -> None: ...
    @overload
    def __init__(self, shape: Sequence[int], arr_if: ArrayInterface) -> None: ...
    def __init__(self, *args, **kwargs):
        """
        TODO! rewrite all the docs for this class
        """
        self._seq_shape, self._arr_if = get_shape_and_array_interface(*args, **kwargs, allow_fixed_shape=False)

        self._buff = None
        self._n_consumed = 0
        self._input_closed = False
        # TODO: better default name, or just default to none and handle in repr?
        self._name = kwargs.get("name", "StreamBuffer")

        # If the overload with the array was used, feed it
        if torch.is_tensor(args[0]) or isinstance(args[0], np.ndarray):
            self.feed(args[0])

    @classmethod
    def new_zeros(cls, *cons_args, seq_size: int, **cons_kwargs) -> "StreamBuffer":
        """
        Returns a StreamBuffer of the given size, filled with zeros.
        """
        buff = cls(*cons_args, **cons_kwargs)
        shape = get_shape_for_seq_size(buff._seq_shape, seq_size)
        buff.feed(buff._arr_if.new_zeros(shape))
        return buff

    @classmethod
    def randn(cls, *cons_args, seq_size: int, **cons_kwargs) -> "StreamBuffer":
        """
        Sample a StreamBuffer of the given size from a normal distribution (discretized for integer types).
        """
        buff = cls(*cons_args, **cons_kwargs)
        shape = get_shape_for_seq_size(buff._seq_shape, seq_size)
        buff.feed(buff._arr_if.new_randn(shape))
        return buff

    def copy(self) -> "StreamBuffer":
        """
        Returns a copy of this StreamBuffer with its own data and n_consumed set 
        """
        buff = StreamBuffer(self._seq_shape, self._arr_if)
        buff.feed(self._arr_if.copy(self._buff))
        return buff

    @property
    def data(self) -> SeqArrayLike | None:
        """
        The data currently in the buffer. None if no data has been fed.
        """
        return self._buff

    @overload
    def __getitem__(self, idx: int) -> SeqArrayLike: ...

    @overload
    def __getitem__(self, sli: slice) -> SeqArrayLike: ...

    def __getitem__(self, sli: int | slice) -> SeqArrayLike:
        """
        TODO: doc
        Reads a sequence of size up to n from the start of buffer without consuming it. If the buffer does not have
        enough elements, the entire buffer is returned.

        :param n: Number of elements to peek at. If None, peeks at the entire buffer.
        :return: The first n elements of the buffer
        """
        if not isinstance(sli, slice):
            sli = slice(sli, sli + 1)
        sli = slice(*sli.indices(self.size))
        assert sli.stop >= sli.start, (
            f"Trying to read {sli.stop - sli.start} elements from {self._name}, n must be positive"
        )

        # If we're reading the entire buffer, just return it
        if sli.stop - sli.start >= self.size:
            # TODO! sort this empty buff thing...
            if self._buff is None:
                return self.spec.empty()
            return self._buff

        # Slice the buffer to make a copy of the elements, so as not to hold a view containing the ones we don't need
        # TODO: settle on copying or not
        return self._arr_if.copy(self._arr_if.get_along_dim(self._buff, sli, dim=self.dim))

    @overload
    def __setitem__(self, idx: int, value: SeqArrayLike) -> None: ...

    @overload
    def __setitem__(self, sli: slice, value: SeqArrayLike) -> None: ...

    def __setitem__(self, sli: int | slice, value: SeqArrayLike) -> None:
        """
        TODO: doc
        Reads a sequence of size up to n from the start of buffer without consuming it. If the buffer does not have
        enough elements, the entire buffer is returned.

        :param n: Number of elements to peek at. If None, peeks at the entire buffer.
        :return: The first n elements of the buffer
        """
        if not isinstance(sli, slice):
            sli = slice(sli, sli + 1)
        sli = slice(*sli.indices(self.size))
        assert sli.stop >= sli.start, (
            f"Trying to set {sli.stop - sli.start} elements from {self._name}, n must be positive"
        )

        self._arr_if.set_along_dim(self._buff, sli, self.dim, value)

    @property
    def n_consumed(self) -> int:
        """
        Number of elements consumed from the buffer alongside the sequence dimension.
        """
        return self._n_consumed

    # TODO: rename seqdim?
    @property
    def dim(self) -> int:
        """
        The dimension along which this buffer is buffering tensors or arrays
        """
        return self.spec.seq_dim

    @property
    def size(self) -> int:
        """
        The available size of the buffer, equivalent to self.shape[self.dim] if the spec is complete.
        TODO: __len__ override? Might be confusing with equivalent tensor len override that returns the size of the
        first dimension.
        """
        return self.shape[self.dim] if self._buff is not None else 0

    @property
    def shape(self) -> Optional[Tuple[int]]:
        """
        The shape of the buffer. None if no data has been fed.
        """
        return self._arr_if.get_shape(self._buff) if self._buff is not None else None

    @property
    def ndim(self) -> int:
        return self.spec.ndim

    @property
    def name(self) -> str:
        """
        Name of this instance if any was given, otherwise the class name
        """
        return self._name or self.__class__.__name__

    @property
    def input_closed(self) -> bool:
        """
        Whether this buffer is closed for input. When a buffer is closed, it will raise on new feed() calls.
        Once a buffer is closed for input, it cannot be reopened.
        """
        return self._input_closed

    @property
    def output_closed(self) -> bool:
        """
        Whether this buffer is closed for output. When true, it will no longer be able to return any data.
        Once a buffer is closed for output, it cannot be reopened.
        If a buffer is closed for output, it necessarily implies that it is also closed for input
        """
        return (not self.size) and self.input_closed

    def close_input(self):
        """
        Closes this buffer for input, i.e. it will no longer accept new feed() calls.
        Once a buffer is closed for input, it cannot be reopened.
        """
        assert not self._input_closed, f"Trying to close input on {self.name}, but input is already closed"
        self._input_closed = True

    def _clear_buf(self):
        if self._buff is not None:
            self._buff = self.spec.new_empty()

    def feed(self, x: SeqArrayLike | "StreamBuffer", close_input=False):
        """
        Feeds data at the end of the buffer. If the buffer is closed for input, this will raise an exception.

        :param x: The data to feed into the buffer. Either a torch of numpy array. It must match the specification
        given in the constructor.
        :param close_input: Whether to close the buffer for input after this call.
        """
        assert not self._input_closed, f"Trying to feed data into {self.name}, but input is closed"

        x = x.data if isinstance(x, StreamBuffer) else x
        is_matching, reason = self.spec.matches(x)
        if not is_matching:
            raise ValueError(f"Cannot feed {type(x)} to {self.name}: {reason}")

        if self._buff is None:
            self._buff = self._arr_if.copy(x)
        else:
            self._buff = self._arr_if.concat(self._buff, x, dim=self.dim)

        if close_input:
            self.close_input()

    def drop(self, n: Optional[int] = None) -> int:
        """
        Removes the first n elements from the buffer. If the buffer does not have enough elements, the entire buffer
        is dropped.

        :param n: Positive number of elements to drop. If None, drops all elements.
        :return: The number of elements dropped
        """
        n = self.size if n is None else n
        assert n >= 0, f"Trying to drop {n} elements from {self._name}, n must be positive"

        # No-op if zero elements to drop
        if n == 0:
            return 0

        # If we're dropping the entire buffer, just clear it
        if n >= self.size:
            out_size = self.size
            self._n_consumed += out_size
            self._clear_buf()
            return out_size

        # Slice the buffer to make a copy of the remaining elements, so as not to hold a view containing the
        # dropped ones
        # TODO: copy
        self._buff = self._arr_if.copy(self[n:])
        self._n_consumed += n

        return n

    def drop_to(self, n: int) -> int:
        """
        Removes elements until n remain. No op if less than n are remaining
        """
        return self.drop(max(self.size - n, 0))

    def read(self, n: Optional[int] = None) -> "StreamBuffer":
        """
        Reads a sequence of size up to n from the start of buffer while dropping it from the buffer. If the
        buffer does not have enough elements, the entire buffer is returned.
        """
        out = self[:n]
        self.drop(n)
        return out

    @classmethod
    def apply(
        cls,
        trsfm: Callable,
        in_seq: "StreamBuffer",
        out_spec: SeqSpec | None = None,
        zero_size_exception_signatures: Iterable[Exception | ExceptionWithSubstring] = DEFAULT_ZERO_SIZE_EXCEPTIONS,
    ) -> "StreamBuffer":
        # TODO! doc
        out_spec = out_spec or in_seq
        out_spec = out_spec.spec if isinstance(out_spec, StreamBuffer) else out_spec

        with torch.inference_mode():
            try:
                with tracer.start_as_current_span(trsfm.__name__ if hasattr(trsfm, "__name__") else "transform"):
                    out_arr = trsfm(in_seq.data)
            except Exception as e:
                if not matches_any_exception(e, zero_size_exception_signatures):
                    raise e
                out_arr = StreamBuffer.empty(out_spec)

            out_seq = out_arr if isinstance(out_arr, StreamBuffer) else cls(out_spec, out_arr, close_input=True)

        return out_seq

    def __repr__(self) -> str:
        return f"<Sequence shape={self.shape} dim={self.dim}>"
