from torchstream.exception_signature import (
    DEFAULT_ZERO_SIZE_EXCEPTIONS,
    matches_any_exception,
)
from torchstream.patching.call_intercept import make_exit_early, intercept_calls
from torchstream.sequence.array_interface import ArrayInterface, NumpyArrayInterface, SeqArray, TensorInterface
from torchstream.sequence.dtype import DeviceLike, SeqArrayLike, SeqDTypeLike, dtypes_compatible, seqdtype, to_seqdtype
from torchstream.sequence.seq_spec import SeqSpec
from torchstream.sequence.sequence import Sequence
from torchstream.sliding_window.sliding_window_params import SlidingWindowParams
from torchstream.sliding_window.sliding_window_params_solver import (
    MaximumSequenceSizeReachedError,
    SlidingWindowParamsSolver,
    find_sliding_window_params,
)
from torchstream.sliding_window.sliding_window_stream import IncorrectSlidingWindowParametersError, SlidingWindowStream
from torchstream.sliding_window.transforms import overlap_windows, run_sliding_window, view_as_windows
from torchstream.stream import NotEnoughInputError, Stream
from torchstream.stream_equivalence import test_stream_equivalent

__all__ = [
    "Sequence",
    "Stream",
    "SlidingWindowParams",
    "SlidingWindowParamsSolver",
    "find_sliding_window_params",
    "test_stream_equivalent",
    "SeqSpec",
    "SlidingWindowStream",
    "intercept_calls",
    "make_exit_early",
    "ArrayInterface",
    "TensorInterface",
    "NumpyArrayInterface",
    "SeqArray",
    "SeqDTypeLike",
    "SeqArrayLike",
    "DeviceLike",
    "seqdtype",
    "to_seqdtype",
    "dtypes_compatible",
    "DEFAULT_ZERO_SIZE_EXCEPTIONS",
    "matches_any_exception",
    "NotEnoughInputError",
    "MaximumSequenceSizeReachedError",
    "IncorrectSlidingWindowParametersError",
    "view_as_windows",
    "overlap_windows",
    "run_sliding_window",
]
