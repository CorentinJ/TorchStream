from typing import Optional, Tuple

import numpy as np

from torchstream.sequence.seq_specs import SeqSpec
from torchstream.sequence.sequence import Sequence


class Container:
    def __init__(self, spec: SeqSpec):
        self.data = None

    @property
    def shape(self) -> Optional[Tuple[int]]:
        """
        The shape of the buffer. None if no data has been fed.
        """
        return self._buff.shape if self._buff is not None else None

    def _clear_buf(self):
        if self._buff is not None:
            new_shape = list(self._buff.shape)
            new_shape[self.dim] = 0

            if self.spec.is_numpy:
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
        is_matching, reason = self.spec.matches(x)
        if not is_matching:
            raise ValueError(f"Cannot feed {type(x)} to {self.name}: {reason}")

        if self._buff is None:
            self._buff = x.clone() if self.spec.is_torch else x.copy()
        else:
            assert (
                self._buff.shape[: self.dim] == x.shape[: self.dim]
                and self._buff.shape[self.dim + 1 :] == x.shape[self.dim + 1 :]
            ), (
                f"Trying to feed {x.shape} data into {self.name}, but the buffer with dim={self.dim} already "
                f"has shape {self._buff.shape}"
            )

            concat_fn = np.concatenate if self.spec.is_numpy else torch.cat
            self._buff = concat_fn((self._buff, x), axis=self.dim)

        if close_input:
            self.close_input()

    def _copy_slice(self, start: Optional[int], stop: Optional[int]) -> Sequence:
        """
        Copies a slice of the buffer to a new tensor
        """
        copy_fn = np.copy if self.spec.is_numpy else torch.clone
        slices = self.spec.get_slices(start, stop)
        return copy_fn(self._buff[slices])

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
        self._buff = self._copy_slice(n, None)

        return n

    def peek(self, n: Optional[int] = None) -> Sequence:
        """
        Reads a sequence of size up to n from the start of buffer without consuming it. If the buffer does not have
        enough elements, the entire buffer is returned.

        :param n: Number of elements to peek at. If None, peeks at the entire buffer.
        :return: The first n elements of the buffer
        """
        n = self.size if n is None else n
        assert n >= 0, f"Trying to peek at {n} elements from {self._name}, n must be positive"

        # If we're reading the entire buffer, just return it
        if n >= self.size:
            if self._buff is None:
                try:
                    return self.spec.empty()
                except ValueError:
                    raise RuntimeError(
                        f"Cannot peek at {self._name} because it has never held data before and the sequence "
                        f"specification is not complete enough to create an empty buffer"
                    )
            return self._buff

        # Slice the buffer to make a copy of the first n elements, so as not to hold a view containing the
        # ones we don't need
        return self._copy_slice(0, n)

    def read(self, n: Optional[int] = None) -> Sequence:
        """
        Reads a sequence of size up to n from the start of buffer while dropping it from the buffer. If the
        buffer does not have enough elements, the entire buffer is returned.
        """
        out = self.peek(n)
        self.drop(n)
        return out

    def get_seq_size(self, seq: Sequence) -> int:
        """
        Returns the size of the sequence dimension in the given sequence. If the sequence does not match the
        specification, raises an error.
        """
        matches, msg = self.matches(seq)
        if not matches:
            raise ValueError(f"Failed to get sequence size: {msg}")

        return seq.shape[self.seq_dim]

    def get_shape_for_seq_size(self, seq_size: int) -> Tuple[int, ...]:
        if seq_size < 0:
            raise ValueError(f"Sequence size must be non-negative, got {seq_size}")
        if self.shape is None or any(dim is None for dim in self.shape):
            raise ValueError(
                f"Cannot sample from a sequence specification with unknown dimensions. Shape is {self.shape}"
            )

        shape = list(self.shape)
        shape[self.seq_dim] = seq_size
        return tuple(shape)

    def get_slices(self, seq_start: int = None, seq_stop: int = None, seq_step: int = None) -> Tuple[slice, ...]:
        """
        Returns a tuple of slices suitable for indexing a numpy or torch tensor in order to obtain the given
        range across the sequence dimension, and the full space across other dimensions.
        """
        slices = [slice(None)] * self.ndim
        slices[self.seq_dim] = slice(seq_start, seq_stop, seq_step)
        return tuple(slices)

    def randn(self, seq_size: int) -> Sequence:
        """
        Sample a sequence of the given size from a normal distribution (discretized for integer types).
        """
        shape = self.get_shape_for_seq_size(seq_size)
        if self.is_torch:
            return torch.randn(shape, dtype=self.dtype, device=self.device)
        else:
            return np.random.randn(*shape).astype(self.dtype)

    def empty(self, seq_size: int = 0) -> Sequence:
        """
        Returns an empty sequence (i.e. uninitialized data) of the given size.
        """
        shape = self.get_shape_for_seq_size(seq_size)
        if self.is_torch:
            return torch.empty(shape, dtype=self.dtype, device=self.device)
        else:
            return np.empty(shape, dtype=self.dtype)


class ArrayContainer:
    def __init__(self, spec: SeqSpec):
        self.data = None

    @property
    def shape(self) -> Optional[Tuple[int]]:
        """
        The shape of the buffer. None if no data has been fed.
        """
        return self._buff.shape if self._buff is not None else None

    def _clear_buf(self):
        if self._buff is not None:
            new_shape = list(self._buff.shape)
            new_shape[self.dim] = 0

            if self.spec.is_numpy:
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
        is_matching, reason = self.spec.matches(x)
        if not is_matching:
            raise ValueError(f"Cannot feed {type(x)} to {self.name}: {reason}")

        if self._buff is None:
            self._buff = x.clone() if self.spec.is_torch else x.copy()
        else:
            assert (
                self._buff.shape[: self.dim] == x.shape[: self.dim]
                and self._buff.shape[self.dim + 1 :] == x.shape[self.dim + 1 :]
            ), (
                f"Trying to feed {x.shape} data into {self.name}, but the buffer with dim={self.dim} already "
                f"has shape {self._buff.shape}"
            )

            concat_fn = np.concatenate if self.spec.is_numpy else torch.cat
            self._buff = concat_fn((self._buff, x), axis=self.dim)

        if close_input:
            self.close_input()

    def _copy_slice(self, start: Optional[int], stop: Optional[int]) -> Sequence:
        """
        Copies a slice of the buffer to a new tensor
        """
        copy_fn = np.copy if self.spec.is_numpy else torch.clone
        slices = self.spec.get_slices(start, stop)
        return copy_fn(self._buff[slices])

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
        self._buff = self._copy_slice(n, None)

        return n

    def peek(self, n: Optional[int] = None) -> Sequence:
        """
        Reads a sequence of size up to n from the start of buffer without consuming it. If the buffer does not have
        enough elements, the entire buffer is returned.

        :param n: Number of elements to peek at. If None, peeks at the entire buffer.
        :return: The first n elements of the buffer
        """
        n = self.size if n is None else n
        assert n >= 0, f"Trying to peek at {n} elements from {self._name}, n must be positive"

        # If we're reading the entire buffer, just return it
        if n >= self.size:
            if self._buff is None:
                try:
                    return self.spec.empty()
                except ValueError:
                    raise RuntimeError(
                        f"Cannot peek at {self._name} because it has never held data before and the sequence "
                        f"specification is not complete enough to create an empty buffer"
                    )
            return self._buff

        # Slice the buffer to make a copy of the first n elements, so as not to hold a view containing the
        # ones we don't need
        return self._copy_slice(0, n)

    def read(self, n: Optional[int] = None) -> Sequence:
        """
        Reads a sequence of size up to n from the start of buffer while dropping it from the buffer. If the
        buffer does not have enough elements, the entire buffer is returned.
        """
        out = self.peek(n)
        self.drop(n)
        return out

    def get_seq_size(self, seq: Sequence) -> int:
        """
        Returns the size of the sequence dimension in the given sequence. If the sequence does not match the
        specification, raises an error.
        """
        matches, msg = self.matches(seq)
        if not matches:
            raise ValueError(f"Failed to get sequence size: {msg}")

        return seq.shape[self.seq_dim]

    def get_shape_for_seq_size(self, seq_size: int) -> Tuple[int, ...]:
        if seq_size < 0:
            raise ValueError(f"Sequence size must be non-negative, got {seq_size}")
        if self.shape is None or any(dim is None for dim in self.shape):
            raise ValueError(
                f"Cannot sample from a sequence specification with unknown dimensions. Shape is {self.shape}"
            )

        shape = list(self.shape)
        shape[self.seq_dim] = seq_size
        return tuple(shape)

    def get_slices(self, seq_start: int = None, seq_stop: int = None, seq_step: int = None) -> Tuple[slice, ...]:
        """
        Returns a tuple of slices suitable for indexing a numpy or torch tensor in order to obtain the given
        range across the sequence dimension, and the full space across other dimensions.
        """
        slices = [slice(None)] * self.ndim
        slices[self.seq_dim] = slice(seq_start, seq_stop, seq_step)
        return tuple(slices)

    def randn(self, seq_size: int) -> Sequence:
        """
        Sample a sequence of the given size from a normal distribution (discretized for integer types).
        """
        shape = self.get_shape_for_seq_size(seq_size)
        if self.is_torch:
            return torch.randn(shape, dtype=self.dtype, device=self.device)
        else:
            return np.random.randn(*shape).astype(self.dtype)

    def empty(self, seq_size: int = 0) -> Sequence:
        """
        Returns an empty sequence (i.e. uninitialized data) of the given size.
        """
        shape = self.get_shape_for_seq_size(seq_size)
        if self.is_torch:
            return torch.empty(shape, dtype=self.dtype, device=self.device)
        else:
            return np.empty(shape, dtype=self.dtype)
