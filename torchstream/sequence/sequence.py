import logging
from typing import Callable, Iterable, Optional, Tuple, overload
from typing import Sequence as _Sequence

import numpy as np
import torch
from opentelemetry import trace

from torchstream.exception_signature import DEFAULT_ZERO_SIZE_EXCEPTIONS, ExceptionWithSubstring
from torchstream.sequence.dtype import DeviceLike, SeqArray, SeqArrayLike, SeqDTypeLike
from torchstream.sequence.seq_spec import SeqSpec

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)


# TODO? Type with a typevar for arrays instead


class Sequence:
    """
    A Sequence is a container for one or multiple arrays (torch tensors or numpy arrays) that each hold sequential data
    along a specified dimension (the sequence dimension).

    For instance, this Sequence could represent stereo audio (encoded as floating-point) and a per-sample Voice
    Activity Detection flag.
    >>> Sequence((2, -1, np.float32), (-1, bool))
    Sequence of size 0 with SeqSpec(
        Array #0: (2, -1) np.float32
        Array #1: (-1,) np.bool
    )

    A Sequence acts as a convenient wrapper around arrays to expose methods relevant to streaming. It does so
    independently of the underlying array type or device, so that implementation of streaming logic can be agnostic
    of these details.

    In the sequence specification required in the constructor, shapes must have a negative value that represents the
    sequence dimension. The dimension can grow and shrink as data is fed and dropped from the Sequence, while the
    sizes of the other dimensions will remain fixed. Array in the specification can take different scales, indicated
    by the absolute value of the negative dimension.

    For instance, this Sequence could represent a 60fps video feed (encoded as uint8 RGB images) along with 48kHz
    mono audio:
    >>> Sequence((-1, 1080, 1920, 3, torch.uint8, "cuda"), (-800, "cuda"))
    Sequence of size 0 with SeqSpec(
        Tensor #0: (-1, 1080, 1920, 3) cuda torch.uint8
        Tensor #1: (-800,) cuda torch.float32
    )
    The scale of the audio array is 800, since for each video frame (1/60s), there are 800 audio samples (1/48000s).

    :param specs: same argument specification as SeqSpec's constructor. Can also be a single SeqSpec directly. If all
    array specifications are given using existing arrays, they will be used to initialize the Sequence's buffers.

    Examples:
    >>> Sequence(3, -1, torch.float32, "cpu")
    >>> Sequence((3, -1), torch.float32, "cpu")
    >>> Sequence(torch.randn(3, 10, 12), seq_dim=1)
    >>> Sequence((torch.randn(3, 10, 12), 1), (torch.randn(10), 0))
    >>> Sequence(SeqSpec(3, -1, torch.float32, "cpu"))
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
        Returns a Sequence of the given size, filled with zeros. Arguments must match the constructor's signature.
        """
        seq = cls(*cons_args, **cons_kwargs)
        seq.feed(*seq.spec.new_zeros_arrays(seq_size))
        return seq

    @classmethod
    def new_randn(cls, *cons_args, seq_size: int, **cons_kwargs) -> "Sequence":
        """
        Sample a Sequence of the given size from a normal distribution (discretized for integer types). Arguments must
        match the constructor's signature.
        """
        seq = cls(*cons_args, **cons_kwargs)
        seq.feed(*seq.spec.new_randn_arrays(seq_size))
        return seq

    def copy(self) -> "Sequence":
        """
        Returns a new Sequence of the same specification with all buffers copied.
        """
        buff = Sequence(self.spec)
        if self._buffs is not None:
            buff.feed(*self._buffs)
        return buff

    @property
    def shapes(self) -> Tuple[Tuple[int, ...], ...]:
        """
        The current shapes of the buffers. Note that this always returns a tuple of shapes, even if there is only
        one array in the specification.
        """
        if self._buffs is None:
            return self.spec.get_shapes_for_seq_size(0)
        return tuple(arr_if.get_shape(buff) for buff, arr_if in zip(self._buffs, self._arr_ifs))

    @property
    def seq_shapes(self) -> Tuple[Tuple[int, ...], ...]:
        """
        Returns the sequence shapes of the specification i.e. the shapes with the sequence dimension set to a
        negative value. Note that this always returns a tuple of shapes, even if there is only one array in the
        specification.
        """
        return self._seq_shapes

    @property
    def seq_dims(self) -> Tuple[int, ...]:
        """
        The indices of the sequence dimensions for each buffer.
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
        The available size of the sequence. For arrays with a sequence scale greater than 1, the sequence size is
        exactly their original size divided by their sequence scale. All arrays in the sequence have the same
        sequence size.
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
    def data(self) -> Tuple[SeqArray, ...]:
        """
        The data currently in the buffer. Returns zero-sized arrays if no data has been fed
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
        Reads from the sequence without consuming it, returning a new Sequence with a copy of the sliced data. The
        indexing is done across the sequence dimension, and across this dimension only (multi indexing is not allowed).

        Note that unlike in torch and numpy, indexing a single element does not squeeze the dimension. Indeed the
        sliced arrays must still conform to the SeqSpec of the Sequence.

        If the specification has a sequence scale greater than 1, slice indices are scaled accordingly.

        Negative indices are supported.

        >>> seq = Sequence.new_zeros(3, -1, 12, seq_size=20)
        >>> seq[5:10]
        Sequence of size 5 with SeqSpec(
            Tensor #0: (3, -1, 12) cpu torch.float32
        )
        >>> seq[3]
        Sequence of size 1 with SeqSpec(
            Tensor #0: (3, -1, 12) cpu torch.float32
        )
        >>> seq = Sequence.new_zeros(3, -2, 12, seq_size=20)
        >>> seq[2].data[0].shape
        (3, 2, 12)
        """
        if isinstance(sli, tuple):
            raise ValueError(
                "Multi-dimensional indexing is not supported. "
                "Sequences can only be indexed along the sequence dimension. Retrieve the sequence's data to index "
                "across other dimensions."
            )

        if not isinstance(sli, slice):
            sli = slice(sli, sli + 1)
        sli = slice(*sli.indices(self.size))

        if self._buffs is None:
            return Sequence(self.spec)

        # Slice the buffer to make a copy of the elements, so as not to hold a view containing the ones we don't need
        out = []
        for buff, seq_dim, scale, arr_if in zip(self._buffs, self.seq_dims, self.seq_scales, self._arr_ifs):
            scaled_sli = slice(
                sli.start * scale if sli.start is not None else None,
                sli.stop * scale if sli.stop is not None else None,
            )
            sliced_array = arr_if.get_along_dim(buff, scaled_sli, seq_dim)
            # NOTE: no need to copy, copy is done in the constructor below
            out.append(sliced_array)
        return self.spec.new_sequence_from_data(*out)

    @overload
    def __setitem__(self, idx: int, value: SeqArrayLike) -> None: ...
    @overload
    def __setitem__(self, sli: slice, value: SeqArrayLike) -> None: ...
    def __setitem__(self, sli: int | slice, value: SeqArrayLike) -> None:
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
        Concatenates arrays at the end of the buffers. The given input must match the sequence specification.
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
        Removes the first n elements from the buffers. For sequence specifications with sequence scales greater than 1,
        n is scaled accordingly.

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
        Removes elements from the buffers until the sequence is of size n. No op if less than n are remaining
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
            - Checking that the output arrays match the given output specification (or this sequence's specification
            if none is provided), raising otherwise
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
    # TODO! settle on whether these methods should exist
    # def stream_apply_iter(
    #     self,
    #     trsfm: Callable,
    #     sli_params: SlidingWindowParams,
    #     chunk_size: int,
    #     out_spec: "SeqSpec | None" = None,
    # ) -> Iterator["Sequence"]:
    #     # TODO! doc
    #     from torchstream.sliding_window.sliding_window_stream import SlidingWindowStream

    #     stream = SlidingWindowStream(trsfm, sli_params, self.spec, out_spec)
    #     yield from stream.forward_in_chunks_iter(self, chunk_size=chunk_size)

    # def stream_apply(
    #     self,
    #     trsfm: Callable,
    #     sli_params: SlidingWindowParams,
    #     chunk_size: int,
    #     out_spec: "SeqSpec | None" = None,
    # ) -> "Sequence":
    #     # TODO! doc
    #     from torchstream.sliding_window.sliding_window_stream import SlidingWindowStream

    #     stream = SlidingWindowStream(trsfm, sli_params, self.spec, out_spec)
    #     return stream.forward_in_chunks(self, chunk_size=chunk_size)

    def __repr__(self) -> str:
        return f"Sequence of size {self.size} with {self.spec}"
