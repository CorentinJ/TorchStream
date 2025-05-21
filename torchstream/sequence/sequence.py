import numbers
from typing import Callable, Optional, Tuple, overload

from torchstream.sequence.array_interface import ArrayInterface
from torchstream.sequence.dtype import SeqArrayLike
from torchstream.sequence.seq_spec import SeqSpec


class Sequence:
    """
    FIXME! rewrite this doc
    Tensor-like class for buffering multidimensional sequential data. Supports both torch tensors and numpy arrays.

    The Sequence class is essentially the implementation of queues for multidimensional tensors. Tensors of the
    same shape (aside from the sequence dimension) are queued up into the buffer and can be read or dropped in
    FIFO order.

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

    # Sigs:
    #   - Spec
    #   - Spec + data (+close) -> check data
    #   - dim + data (+close) -> derive spec
    #   - Seq (possibly multiple +close) -> copy data
    # close & name forced in kwarg

    def __init__(self, *args, **kwargs):
        """
        TODO! rewrite all the docs for this class

        :param data: optional initial tensors to buffer
        :param dim: data specification for the sequence, containing at the minimum the shape or sequence dimension.
        :param name: a name to give to this instance, useful for debugging
        """
        # TODO: verify compatible types when multiple inputs provided
        if isinstance(args[0], SeqSpec) or "spec" in kwargs:
            # Case 1 & 2
            self.spec = kwargs.get("spec", args[0])
            arrays = args[1:]
        elif isinstance(args[0], numbers.Number) and len(args) > 1:
            # Case 3
            # TODO: implement that same overload for SeqSpec
            shape = ArrayInterface(args[1]).get_shape(args[1])
            shape[args[0]] = -1
            self.spec = SeqSpec(shape, args[1])
            arrays = args[1:]
        elif isinstance(args[0], Sequence):
            # Case 4
            self.spec = args[0].spec
            arrays = args
        else:
            raise TypeError(f"Cannot infer a SeqSpec from positional arguments Sequence{args}")

        self._arr_if = self.spec._arr_if
        self._buff = None
        self._n_consumed = 0
        self._input_closed = False
        # TODO: better default name, or just default to none and handle in repr?
        self._name = kwargs.get("name", "Sequence")

        for arr in arrays:
            # TODO! copy (only if sequence)
            self.feed(arr)
        if kwargs.get("close_input", False):
            self.close_input()

    @classmethod
    def empty(cls, seq_spec: SeqSpec, seq_size: int = 0) -> "Sequence":
        """
        Returns an empty Sequence of the given shape. The array's values are uninitialized.
        """
        return cls(seq_spec, seq_spec.new_empty(seq_size))

    @classmethod
    def zeros(cls, seq_spec: SeqSpec, seq_size: int) -> "Sequence":
        """
        Returns a Sequence of the given size, filled with zeros.
        """
        return cls(seq_spec, seq_spec.new_zeros(seq_size))

    @classmethod
    def randn(cls, seq_spec: SeqSpec, seq_size: int) -> "Sequence":
        """
        Sample a Sequence of the given size from a normal distribution (discretized for integer types).
        """
        return cls(seq_spec, seq_spec.new_randn(seq_size))

    def copy(self) -> "Sequence":
        """
        Returns a deep copy of this Sequence.
        """
        return Sequence(self.spec, self._arr_if.copy(self._buff), name=self._name)

    @property
    def data(self) -> SeqArrayLike:
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
    def consumed(self) -> int:
        """
        Number of elements consumed from the buffer alongside the sequence dimension.
        """
        return self._n_consumed

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

    def feed(self, x: SeqArrayLike | "Sequence", close_input=False):
        """
        Feeds data at the end of the buffer. If the buffer is closed for input, this will raise an exception.

        :param x: The data to feed into the buffer. Either a torch of numpy array. It must match the specification
        given in the constructor.
        :param close_input: Whether to close the buffer for input after this call.
        """
        assert not self._input_closed, f"Trying to feed data into {self.name}, but input is closed"

        x = x.data if isinstance(x, Sequence) else x
        is_matching = self.spec.matches(x)
        if not is_matching:
            # TODO!
            raise ValueError(f"Cannot feed {type(x)} to {self.name}: reason")

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

    def consume(self, n: Optional[int] = None) -> "Sequence":
        """
        Reads a sequence of size up to n from the start of buffer while dropping it from the buffer. If the
        buffer does not have enough elements, the entire buffer is returned.
        """
        out = self[:n]
        self.drop(n)
        return out

    # TODO: multiple inputs/outputs support
    @classmethod
    def apply(cls, trsfm: Callable, in_seq: "Sequence", out_spec: SeqSpec | None = None) -> "Sequence":
        out_spec = out_spec or in_seq
        out_spec = out_spec.spec if isinstance(out_spec, Sequence) else out_spec

        out_arr = trsfm(in_seq.data)
        out_seq = cls(out_spec, out_arr, close_input=True)
        return out_seq

    def __repr__(self) -> str:
        return f"<Sequence shape={self.shape} dim={self.dim}>"
