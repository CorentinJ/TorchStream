from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Iterable, Optional, Tuple, Union, overload
from typing import Sequence as _Sequence

import numpy as np
import torch
from opentelemetry import trace

from torchstream.exception_signature import DEFAULT_ZERO_SIZE_EXCEPTIONS, ExceptionWithSubstring, matches_any_exception
from torchstream.sequence.array_interface import SeqArray, TensorInterface
from torchstream.sequence.dtype import DeviceLike, SeqArrayLike, SeqDTypeLike
from torchstream.sequence.sequential_array import (
    array_matches_shape_and_type,
    get_shape_and_array_interface,
    get_shape_for_seq_size,
)

# We have a circular dependecy for a few convenience methods
if TYPE_CHECKING:
    from torchstream.sequence.sequence import Sequence

tracer = trace.get_tracer(__name__)


class SeqSpec:
    """
    A SeqSpec (sequence specification) describes the data format of one or multiple arrays (torch tensors or numpy
    arrays) that each hold sequential data along a specified dimension (the sequence dimension).

    For instance, this SeqSpec could represent the format for stereo audio (encoded as floating-point) and a
    per-sample Voice Activity Detection flag.
    >>> SeqSpec((2, -1, np.float32), (-1, bool))
    SeqSpec(
        Array #0: (2, -1) np.float32
        Array #1: (-1,) np.bool
    )

    A SeqSpec dictates the shape, type and device (for torch) of array data. It also exposes data validation, data
    manipulation methods and data constructors to work with sequential arrays.

    In the sequence specification required in the constructor, shapes must have a negative value that represents the
    sequence dimension. Arrays can match the SeqSpec's format with varying sizes along the sequence dimension, but
    must match exactly along the other dimensions. The sequence dimension can take different scales, indicated
    by its absolute value.

    For instance, this SeqSpec could represent the data format for a 60fps video feed (encoded as uint8 RGB images)
    along with 48kHz mono audio:
    >>> SeqSpec((-1, 1080, 1920, 3, torch.uint8, "cuda"), (-800, "cuda"))
    SeqSpec(
        Tensor #0: (-1, 1080, 1920, 3) cuda torch.uint8
        Tensor #1: (-800,) cuda torch.float32
    )
    The scale of the audio array is 800, since for each video frame (1/60s), there are 800 audio samples (1/48000s).

    :param specs: the arguments can take the following forms:
    - A shape expressed with multiple integers or as a single tuple of integers, with optional dtype and device
    arguments (their order is interchangeable). The shape must contain exactly one negative integer representing the
    sequence dimension and its scale.
    - A single array (torch tensor or numpy array) with an optional seq_dim argument to specify the sequence dimension.
    Note that it is not possible to specify a sequence scale other than 1 with this argument.
    - Multiple of the aboves, each provided as a tuple. Keyword arguments cannot be used in this case.

    Examples:
    >>> SeqSpec(3, -1, torch.float32, "cpu")
    >>> SeqSpec((3, -1), torch.float32, "cpu")
    >>> SeqSpec(torch.randn(3, 10, 12), seq_dim=1)
    >>> SeqSpec((torch.randn(3, 10, 12), 1), (torch.randn(10), 0))
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
    def __init__(self, *specs, **kwargs) -> None:
        if all(isinstance(spec, tuple) for spec in specs) and not kwargs:
            self.specs = [get_shape_and_array_interface(*spec) for spec in specs]
        else:
            self.specs = [get_shape_and_array_interface(*specs, **kwargs)]

    def __len__(self) -> int:
        """
        Returns the number of arrays in the specification.
        """
        return len(self.specs)

    @property
    def seq_shapes(self) -> Tuple[Tuple[int, ...], ...]:
        """
        Returns the sequence shapes of the specification.
        """
        return tuple(shape for shape, _ in self.specs)

    @property
    def seq_dims(self) -> Tuple[int, ...]:
        """
        Returns the sequence dimensions of all arrays in the specification.
        """
        return tuple(next(i for i, dim in enumerate(shape) if dim < 0) for shape, _ in self.specs)

    @property
    def seq_scales(self) -> Tuple[int, ...]:
        """
        Returns the sequence scales of all arrays in the specification. The sequence scale is the absolute value
        of the sequence dimension
        """
        return tuple(-shape[seq_dim] for shape, seq_dim in zip((shape for shape, _ in self.specs), self.seq_dims))

    def matches(self, *arrs: SeqArrayLike) -> Tuple[bool, str]:
        """
        Returns whether the given arrays are compatible with the sequence specification. Compatible in this context
        means that, at least, all arrays:
            - are each from the same library as the specification (torch, numpy, ...)
            - have the same number representation type (floating point, integer, complex, ...) as their respective
            sequence dtype
            - match the shape of the specification (except for the sequence dimension which is a strictly
            negative integer)

        The arrays must be provided in the same order as the specification.

        :return: A tuple (matches, reason). If matches is False, reason contains a human-readable explanation of
        why the arrays do not match the specification.
        """
        if len(arrs) != len(self.specs):
            try:
                shape_str = " with shapes [" + ", ".join(str(tuple(a.shape)) for a in arrs) + "]"
            except Exception:
                shape_str = ""
            return False, f"specification {self} expects {len(self.specs)} arrays, got {len(arrs)}{shape_str}"

        for idx, (arr, (shape, arr_if)) in enumerate(zip(arrs, self.specs)):
            matches, reason = array_matches_shape_and_type(arr, shape, arr_if)
            if not matches:
                return False, f"array #{idx}: {reason}" if len(self.specs) > 1 else reason

        # TODO: verify sizes match

        return True, ""

    def apply(
        self,
        trsfm: Callable,
        *in_arrs: SeqArrayLike,
        out_spec: Optional["SeqSpec"] = None,
        zero_size_exception_signatures: Iterable[
            Union[Exception, ExceptionWithSubstring]
        ] = DEFAULT_ZERO_SIZE_EXCEPTIONS,
    ) -> Tuple[SeqArrayLike, ...]:
        """
        Forwards the given input arrays through the given transform while:
            - Using torch's inference_mode
            - Checking that the input arrays match this specification, raising otherwise
            - Checking that the output arrays match the given output specification (or this specification if none is
            given), raising otherwise
            - Catching zero-size exceptions raised by the transform to return empty arrays instead

        :param trsfm: A transform that takes in arrays matching exactly this specification, and returning arrays
        matching exactly the output specification.
        :param in_arrs: Input arrays to forward through the transform.
        :param out_spec: Specification that the output arrays must match. If None, it is assumed to be the same as
        this specification.
        :param zero_size_exception_signatures: Signatures of exceptions that indicate that the transform could not
        produce any output due to the input arrays being too small, leading to a zero-size output. You may pass
        an empty iterable to disable this behavior. You can also add to the base set of exceptions
        DEFAULT_ZERO_SIZE_EXCEPTIONS with your own exception signatures.
        :return: Output arrays returned by the transform.
        """
        out_spec = out_spec or self

        matches, reason = self.matches(*in_arrs)
        if not matches:
            raise ValueError(f"Input arrays do not match the input specification: {reason}")

        with torch.inference_mode():
            try:
                with tracer.start_as_current_span(trsfm.__name__ if hasattr(trsfm, "__name__") else "transform"):
                    out_arrs = trsfm(*in_arrs)

                if not isinstance(out_arrs, tuple):
                    out_arrs = (out_arrs,)

            except Exception as e:
                if not matches_any_exception(e, zero_size_exception_signatures):
                    raise e
                out_arrs = out_spec.new_empty_arrays()

        return out_arrs

    def get_shapes_for_seq_size(self, seq_size: int) -> Tuple[Tuple[int, ...], ...]:
        """
        Returns the shapes of all arrays in the specification for a sequence of the given size. Each shape is returned
        with the sequence dimension replaced by the given sequence size. If the sequence dimension is a value other than
        -1, the absolute value of that integer is used as a multiplier for the sequence size.
        Example
        --------
        >>> spec = SeqSpec((-1, 10, 15), (8, -2))
        >>> spec.get_shapes_for_seq_size(14)
        ((14, 10, 15), (8, 28))
        """
        return tuple([get_shape_for_seq_size(shape, seq_size) for shape, _ in self.specs])

    def new_empty_arrays(self, seq_size: int = 0) -> Tuple[SeqArray, ...]:
        """
        Returns empty arrays with the given specification. The array's values are uninitialized.
        """
        shapes = self.get_shapes_for_seq_size(seq_size)
        return tuple(arr_if.new_empty(*shape) for shape, (_, arr_if) in zip(shapes, self.specs))

    def new_zeros_arrays(self, seq_size: int) -> Tuple[SeqArray, ...]:
        """
        Returns arrays of the given sequence size with the given specification, filled with zeros.
        """
        shapes = self.get_shapes_for_seq_size(seq_size)
        return tuple(arr_if.new_zeros(*shape) for shape, (_, arr_if) in zip(shapes, self.specs))

    def new_randn_arrays(self, seq_size: int) -> Tuple[SeqArray, ...]:
        """
        Sample arrays of the given sequence size from a normal distribution (discretized for integer types).
        """
        shapes = self.get_shapes_for_seq_size(seq_size)
        return tuple(arr_if.new_randn(*shape) for shape, (_, arr_if) in zip(shapes, self.specs))

    def new_empty_sequence(self) -> Sequence:
        """
        Returns empty an Sequence with the given specification.
        """
        from torchstream.sequence.sequence import Sequence

        return Sequence(self)

    def new_zero_sequence(self, seq_size: int) -> Sequence:
        """
        Returns a Sequence of the given sequence size with the given specification, buffers filled with zeros.
        """
        from torchstream.sequence.sequence import Sequence

        return Sequence.new_zeros(self, seq_size=seq_size)

    def new_randn_sequence(self, seq_size: int) -> Sequence:
        """
        Sample a Sequence of the given sequence size from a normal distribution (discretized for integer types).
        """
        from torchstream.sequence.sequence import Sequence

        return Sequence.new_randn(self, seq_size=seq_size)

    def new_sequence_from_data(self, *arrs: SeqArrayLike) -> Sequence:
        """
        Creates a new Sequence with the given specification, feeding it the given arrays.
        """
        from torchstream.sequence.sequence import Sequence

        seq = Sequence(self)
        seq.feed(*arrs)
        return seq

    def __repr__(self) -> str:
        out = ""
        for idx, (shape, arr_if) in enumerate(self.specs):
            device_str = f"{arr_if.device} " if hasattr(arr_if, "device") else ""
            dtype_str = str(arr_if.dtype) if not isinstance(arr_if.dtype, np.dtype) else f"np.{arr_if.dtype}"
            array_str = "Tensor" if isinstance(arr_if, TensorInterface) else " Array"
            out += f"\n   {array_str} #{idx}: {shape} {device_str}{dtype_str}"
        return f"SeqSpec({out}\n)"
