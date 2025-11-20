import logging
from typing import Callable, Iterable, Iterator, Optional, Tuple, overload
from typing import Sequence as _Sequence

import numpy as np
import torch
from opentelemetry import trace

from torchstream.exception_signature import DEFAULT_ZERO_SIZE_EXCEPTIONS, ExceptionWithSubstring
from torchstream.sequence.dtype import DeviceLike, SeqArrayLike, SeqDTypeLike
from torchstream.sequence.seq_spec import SeqSpec
from torchstream.sliding_window.sliding_window_params import SlidingWindowParams

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)


# TODO? Type with a typevar for arrays instead


class Sequence:
    """
    FIXME! rewrite this doc
    Tensor-like class for buffering multidimensional sequential data. Supports both torch tensors and numpy arrays.

    >>> buff = Sequence(spec=SeqSpec(seq_dim=1))
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
        self, shape: _Sequence[int], dtype: SeqDTypeLike = torch.float32, device: DeviceLike = "cpu"
    ) -> None: ...
    @overload
    def __init__(self, array: SeqArrayLike, seq_dim: int = -1) -> None: ...
    @overload
    def __init__(self, *specs: Tuple) -> None: ...
    @overload
    def __init__(self, seq_spec: SeqSpec) -> None: ...
    def __init__(self, *args, **kwargs):
        """
        TODO! rewrite all the docs for this class
        """
        if len(args) == 1 and isinstance(args[0], SeqSpec):
            self.spec = args[0]
        else:
            self.spec = SeqSpec(*args, **kwargs)

        self._buffs = None
        self._seq_shapes, self._arr_ifs = zip(*self.spec.specs)

        # If the overload with the array(s) was used, feed them
        if torch.is_tensor(args[0]) or isinstance(args[0], np.ndarray):
            self.feed(args[0])
        elif all(isinstance(arg, tuple) and (torch.is_tensor(arg) or isinstance(arg, np.ndarray)) for arg in args):
            self.feed(*(arg[0] for arg in args))

    @classmethod
    def new_zeros(cls, *cons_args, seq_size: int, **cons_kwargs) -> "Sequence":
        """
        Returns a Sequence of the given size, filled with zeros.
        """
        seq = cls(*cons_args, **cons_kwargs)
        seq.feed(*seq.spec.new_zeros_arrays(seq_size))
        return seq

    @classmethod
    def new_randn(cls, *cons_args, seq_size: int, **cons_kwargs) -> "Sequence":
        """
        Sample a Sequence of the given size from a normal distribution (discretized for integer types).
        """
        seq = cls(*cons_args, **cons_kwargs)
        seq.feed(*seq.spec.new_randn_arrays(seq_size))
        return seq

    def copy(self) -> "Sequence":
        """
        Returns a copy of this Sequence with its own data.
        """
        buff = Sequence(self.spec)
        if self._buffs is not None:
            buff.feed(*self._buffs)
        return buff

    @property
    def shapes(self) -> Tuple[Tuple[int, ...], ...]:
        """
        The current shapes of the buffers
        """
        if self._buffs is None:
            return self.spec.get_shapes_for_seq_size(0)
        return tuple(arr_if.get_shape(buff) for buff, arr_if in zip(self._buffs, self._arr_ifs))

    @property
    def seq_shapes(self) -> Tuple[Tuple[int, ...], ...]:
        """
        Returns the sequence shapes of the specification.
        """
        return self._seq_shapes

    @property
    def seq_dims(self) -> Tuple[int, ...]:
        """
        The dimension along which the buffers are concatenating or reading from tensors or arrays
        """
        return self.spec.seq_dims

    @property
    def seq_scales(self) -> Tuple[int, ...]:
        """
        Returns the sequence scales of all arrays in the specification. The sequence scale is the absolute value
        of the sequence dimension
        """
        return self.spec.seq_scales

    @property
    def size(self) -> int:
        """
        The available size of the sequence
        """
        if self._buffs is None:
            return 0

        size = None
        for buff, seq_dim, seq_scale, arr_if in zip(self._buffs, self.seq_dims, self.seq_scales, self._arr_ifs):
            orig_size = arr_if.get_shape(buff)[seq_dim]
            if size is not None:
                assert size * seq_scale == orig_size
            else:
                size = orig_size // seq_scale
        return size

    @property
    def ndim(self) -> int:
        """
        Number of dimensions, i.e. len(self.shape)
        """
        return len(self.spec)

    @property
    def data(self) -> Tuple[SeqArrayLike, ...]:
        """
        The data currently in the buffer. Returns empty arrays if no data has been fed
        """
        if self._buffs is None:
            return self.spec.new_empty_arrays()
        return self._buffs

    @overload
    def __getitem__(self, idx: int) -> "Sequence": ...
    @overload
    def __getitem__(self, sli: slice) -> "Sequence": ...
    def __getitem__(self, sli: int | slice) -> "Sequence":
        """
        TODO: doc
        Reads a sequence of size up to n from the start of buffer without consuming it. If the buffer does not have
        enough elements, the entire sequence is returned.

        :param n: Number of elements to peek at. If None, peeks at the entire buffers.
        :return: A sequence with the n first elements of the buffers.
        """
        if not isinstance(sli, slice):
            sli = slice(sli, sli + 1)
        sli = slice(*sli.indices(self.size))
        assert sli.stop >= sli.start, (
            f"Trying to read {sli.stop - sli.start} elements from {self._name}, n must be positive"
        )

        # If we're reading the entire buffer, just return self
        if sli.stop - sli.start >= self.size:
            return self.copy()

        # Slice the buffer to make a copy of the elements, so as not to hold a view containing the ones we don't need
        out = []
        for buff, seq_dim, scale, arr_if in zip(self._buffs, self.seq_dims, self.seq_scales, self._arr_ifs):
            scaled_sli = slice(
                sli.start * scale if sli.start is not None else None,
                sli.stop * scale if sli.stop is not None else None,
            )
            sliced_array = arr_if.get_along_dim(buff, scaled_sli, seq_dim)
            out.append(arr_if.copy(sliced_array))
        return self.spec.new_sequence_from_data(*out)

    @overload
    def __setitem__(self, idx: int, value: SeqArrayLike) -> None: ...
    @overload
    def __setitem__(self, sli: slice, value: SeqArrayLike) -> None: ...
    def __setitem__(self, sli: int | slice, value: SeqArrayLike) -> None:
        """ """
        if not isinstance(sli, slice):
            sli = slice(sli, sli + 1)
        sli = slice(*sli.indices(self.size))
        assert sli.stop >= sli.start, (
            f"Trying to set {sli.stop - sli.start} elements from {self._name}, n must be positive"
        )

        for buff, seq_dim, scale, arr_if in zip(self._buffs, self.seq_dims, self.seq_scales, self._arr_ifs):
            scaled_sli = slice(
                sli.start * scale if sli.start is not None else None,
                sli.stop * scale if sli.stop is not None else None,
            )
            arr_if.set_along_dim(buff, scaled_sli, seq_dim, value)

    @overload
    def feed(self, *x: SeqArrayLike) -> None: ...
    @overload
    def feed(self, x: "Sequence") -> None: ...
    def feed(self, *x):
        """
        TODO
        """
        if len(x) == 1 and isinstance(x[0], Sequence):
            x = x[0].data

        matches, reason = self.spec.matches(*x)
        if not matches:
            raise ValueError(f"Cannot feed arrays to Sequence: {reason}")

        # TODO: use arr_if.normalize?
        if self._buffs is None:
            self._buffs = tuple(arr_if.copy(arr) for arr, arr_if in zip(x, self._arr_ifs))
        else:
            self._buffs = tuple(
                arr_if.concat(buff, arr, dim=seq_dim)
                for buff, arr, seq_dim, arr_if in zip(self._buffs, x, self.seq_dims, self._arr_ifs)
            )

    def drop(self, n: Optional[int] = None) -> int:
        """
        Removes the first n elements from the buffer. If the buffer does not have enough elements, the entire buffer
        is dropped.

        :param n: Positive number of elements to drop. If None, drops all elements.
        :return: The number of elements dropped
        """
        n = self.size if n is None else n
        assert n >= 0, f"Trying to drop {n} elements, n must be positive"

        # No-op if zero elements to drop
        if n == 0:
            return 0

        # If we're dropping the entire buffer, just clear it
        if n >= self.size:
            out_size = self.size
            self._buffs = None
            return out_size

        # Slice the buffer to make a copy of the remaining elements, so as not to hold a view containing the
        # dropped ones
        self._buffs = self[n:]._buffs

        return n

    def drop_to(self, n: int) -> int:
        """
        Removes elements from all buffers until they are of size n. No op if less than n are remaining
        """
        return self.drop(max(self.size - n, 0))

    def clear(self) -> int:
        """
        Clears the buffer entirely, returning the number of elements dropped.
        """
        return self.drop()

    def read(self, n: Optional[int] = None) -> "Sequence":
        """
        Reads a sequence of size up to n from the start of buffer while dropping it from the buffer. If the
        buffer does not have enough elements, the entire buffer is returned.
        """
        out = self[:n]
        self.drop(n)
        return out

    def apply(
        self,
        trsfm: Callable,
        out_spec: "SeqSpec | None" = None,
        zero_size_exception_signatures: Iterable[Exception | ExceptionWithSubstring] = DEFAULT_ZERO_SIZE_EXCEPTIONS,
    ) -> "Sequence":
        """
        Forwards the sequence's data (without consuming it) through the given transform while:
            - Using torch's inference_mode
            - Checking that the output arrays match the given output specification (or this specification if none is
            given), raising otherwise
            - Catching zero-size exceptions raised by the transform to return empty arrays instead

        :param trsfm: A transform that takes in arrays matching exactly this sequence's specification (as positional
        arguments), and returning arrays matching exactly the output specification.
        :param out_spec: Specification that the output arrays must match. If None, it is assumed to be the same as
        this specification.
        :param zero_size_exception_signatures: Signatures of exceptions that indicate that the transform could not
        produce any output due to the input arrays being too small, leading to a zero-size output. You may pass
        an empty iterable to disable this behavior. You can also add to the base set of exceptions
        DEFAULT_ZERO_SIZE_EXCEPTIONS with your own exception signatures.
        :return: Output arrays returned by the transform.
        """
        out_spec = out_spec or self.spec

        out_arrs = self.spec.apply(
            trsfm,
            *self.data,
            out_spec=out_spec,
            zero_size_exception_signatures=zero_size_exception_signatures,
        )

        return out_spec.new_sequence_from_data(*out_arrs)

    # TODO: naive equivalents

    def stream_apply_iter(
        self,
        trsfm: Callable,
        sli_params: SlidingWindowParams,
        chunk_size: int,
        out_spec: "SeqSpec | None" = None,
    ) -> Iterator["Sequence"]:
        # TODO! doc
        from torchstream.sliding_window.sliding_window_stream import SlidingWindowStream

        stream = SlidingWindowStream(trsfm, sli_params, self.spec, out_spec)
        yield from stream.forward_in_chunks_iter(self, chunk_size=chunk_size)

    def stream_apply(
        self,
        trsfm: Callable,
        sli_params: SlidingWindowParams,
        chunk_size: int,
        out_spec: "SeqSpec | None" = None,
    ) -> "Sequence":
        # TODO! doc
        from torchstream.sliding_window.sliding_window_stream import SlidingWindowStream

        stream = SlidingWindowStream(trsfm, sli_params, self.spec, out_spec)
        return stream.forward_in_chunks(self, chunk_size=chunk_size)

    def __repr__(self) -> str:
        return f"Sequence of size {self.size} with {self.spec}"
