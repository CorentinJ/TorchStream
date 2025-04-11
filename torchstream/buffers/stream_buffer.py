from typing import Optional, Tuple

import numpy as np
import torch

from torchstream.sequence_spec import SeqSpec, Sequence


class StreamBuffer:
    """
    Tensor-like class for buffering multidimensional sequential data. Supports both torch tensors and numpy arrays.

    The StreamBuffer class is essentially the implementation of queues for multidimensional tensors. Tensors of the
    same shape (aside from the sequence dimension) are queued up into the buffer and can be read or dropped in
    FIFO order.

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

    def __init__(self, *data, spec: SeqSpec, name: str = None):
        """
        :param data: optional initial tensors to buffer
        :param dim: data specification for the sequence, containing at the minimum the shape or sequence dimension.
        :param name: a name to give to this instance, useful for debugging
        """
        self._spec = spec
        self._buff = None
        self._input_closed = False
        self._offset = 0
        self._name = name or "Buffer"

        for x in data:
            self.feed(x)

    @property
    def dim(self) -> int:
        """
        The dimension along which this buffer is buffering tensors or arrays
        """
        return self._spec.seq_dim

    @property
    def size(self) -> int:
        """
        The available size of the buffer, equivalent to self.shape[self.dim] if any data has been fed.
        TODO: __len__ override? Might be confusing with equivalent tensor len override that returns the size of the
        first dimension.
        """
        return self._buff.shape[self.dim] if self._buff is not None else 0

    @property
    def shape(self) -> Optional[Tuple[int]]:
        """
        The shape of the buffer. None if no data has been fed.
        """
        return self._buff.shape if self._buff is not None else None

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
            new_shape = list(self._buff.shape)
            new_shape[self.dim] = 0

            if self._spec.is_numpy:
                self._buff = np.empty_like(self._buff, shape=new_shape)
            else:
                self._buff = self._buff.new_empty(new_shape)

    def feed(self, x: Sequence, close_input=False):
        """
        Feeds data at the end of the buffer. If the buffer is closed for input, this will raise an exception.

        :param x: The data to feed into the buffer. Either a torch of numpy array. It must match the specification
        given in the constructor.
        :param close_input: Whether to close the buffer for input after this call.
        """
        assert not self._input_closed, f"Trying to feed data into {self.name}, but input is closed"
        is_matching, reason = self._spec.matches(x)
        if not is_matching:
            raise ValueError(f"Cannot feed {x} to {self.name}: {reason}")

        if self._buff is None:
            self._buff = x.clone() if self._spec.is_torch else x.copy()
        else:
            assert (
                self._buff.shape[: self.dim] == x.shape[: self.dim]
                and self._buff.shape[self.dim + 1 :] == x.shape[self.dim + 1 :]
            ), (
                f"Trying to feed {x.shape} data into {self.name}, but the buffer with dim={self.dim} already "
                f"has shape {self._buff.shape}"
            )

            concat_fn = np.concatenate if self._spec.is_numpy else torch.cat
            self._buff = concat_fn((self._buff, x), axis=self.dim)

        if close_input:
            self.close_input()

    def _copy_slice(self, sli: slice) -> Sequence:
        """
        Copies a slice of the buffer to a new tensor
        """
        copy_fn = np.copy if self._spec.is_numpy else torch.clone
        slices = [slice(None)] * self._buff.ndim
        slices[self.dim] = sli
        return copy_fn(self._buff[tuple(slices)])

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
            self._clear_buf()
            return out_size

        # Slice the buffer to make a copy of the remaining elements, so as not to hold a view containing the
        # dropped ones
        self._buff = self._copy_slice(slice(n, None))

        return n

    def drop_to(self, n: int) -> int:
        """
        Removes elements until n remain. No op if less than n are remaining
        """
        return self.drop(max(self.size - n, 0))

    def peek(self, n: Optional[int] = None) -> Sequence:
        """
        Reads a sequence of size up to n from the start of buffer without consuming it. If the buffer does not have
        enough elements, the entire buffer is returned.

        :param n: Number of elements to peek at. If None, peeks at the entire buffer. A read of 0 is possible only if
        the buffer has held data in the past.
        :return: The first n elements of the buffer
        """
        n = self.size if n is None else n
        assert n >= 0, f"Trying to peek at {n} elements from {self._name}, n must be positive"

        # If we're reading the entire buffer, just return it
        if n >= self.size:
            if self._buff is None:
                try:
                    return self._spec.empty()
                except ValueError:
                    raise RuntimeError(
                        f"Cannot peek at {self._name} because it has never held data before and the sequence "
                        f"specification is not complete enough to create an empty buffer"
                    )
            return self._buff

        # Slice the buffer to make a copy of the first n elements, so as not to hold a view containing the
        # ones we don't need
        return self._copy_slice(slice(0, n))

    def read(self, n: Optional[int] = None) -> Sequence:
        """
        Reads a sequence of size up to n from the start of buffer while dropping it from the buffer. If the
        buffer does not have enough elements, the entire buffer is returned.
        """
        out = self.peek(n)
        self.drop(n)
        return out

    def __repr__(self) -> str:
        return f"<buffer shape={self.shape} dim={self.dim}>"
