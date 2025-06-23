import math
from typing import Tuple, Union, overload

from z3 import ArithRef, If

from torchstream.sliding_window.sliding_window_params import SlidingWindowParams

IntLike = Union[int, ArithRef]


def ceil_div(a: IntLike, b: IntLike) -> IntLike:
    """Ceiling division that works for both Python ints and z3 expressions."""
    if isinstance(a, int) and isinstance(b, int):
        return int(math.ceil(a / b))
    return (a + b - 1) / b


def max_(a: IntLike, b: IntLike) -> IntLike:
    """max() for both Python ints and z3 expressions."""
    if isinstance(a, int) and isinstance(b, int):
        return max(a, b)
    return If(a > b, a, b)


@overload
def get_streaming_params(sli_params: SlidingWindowParams) -> Tuple[int, int, int, int, int]: ...
@overload
def get_streaming_params(
    k_i: IntLike, s_i: IntLike, p_l: IntLike, p_r: IntLike, k_o: IntLike, s_o: IntLike, t_o: IntLike
) -> Tuple[IntLike, IntLike, IntLike, IntLike, IntLike]: ...
def get_streaming_params(*args):
    """
    Derives parameters necessary for streaming from the sliding window parameters. Multiple sliding window parameters
    can give rise to the same streaming parameters. Also, incorrect sliding window parameters can give rise to correct
    but suboptimal streaming parameters that use too much context.

    This function returns 5 parameters:
    - stride_in: the stride (reduction factor) for the input sequence
    - stride_out: the stride (multiplication factor) for the output sequence
      TODO? rename to "delay"
    - in_offset: offset for the input sequence
    - out_offset: offset for the output sequence
    - in_context_size: number of input elements to be buffered as context

    These parameters offset the effective size of the input sequence and output sequence, and account for
    wasted windows due to padding and trimming, as well as the context needed for overlapping windows.
    """
    if len(args) == 1 and isinstance(args[0], SlidingWindowParams):
        p = args[0]
        k_i, s_i, p_l, p_r, k_o, s_o, t_o = (
            p.kernel_size_in,
            p.stride_in,
            p.left_pad,
            p.right_pad,
            p.kernel_size_out,
            p.stride_out,
            p.out_trim,
        )
    elif len(args) == 7:
        k_i, s_i, p_l, p_r, k_o, s_o, t_o = args
    else:
        raise TypeError("Invalid arguments for get_streaming_params")

    # These parameters offset the effective size of the input sequence
    in_offset = k_i - p_l

    # Out trimming also offsets the output sequence
    out_offset = t_o

    # Number of windows that are wasted on the left solely due to padding. "Wasted" here means that we recompute
    # these windows on each step despite them being unnecessary, simply because the transform re-pads the input
    # every time. If it is possible to remove padding from the transform and manually pad the streamed input,
    # this waste of compute can be avoided.
    # Note that right padding wastes compute just as much, however it does not require any context to be stored.
    n_left_wins_wasted = ceil_div(p_l, s_i)

    # For a given output window, the number of other output windows that overlap it. Only >0 when the out stride
    # is smaller than the out kernel size.
    # Note that we need to buffer enough past context in order to have the overlapping windows necessary in
    # computing a given output. This induces redundant compute that could be avoided if the reduce operation on
    # overlapping windows (e.g. a vector sum) is known.
    n_overlapping_out_wins = ceil_div(k_o, s_o) - 1

    # Extra windows necessary to make up for windows lost on the left due to output trimming
    n_trimmed_wins = ceil_div(t_o, s_o)

    # Number of windows that are needed as context
    windows_context_size = n_left_wins_wasted + max_(n_overlapping_out_wins, n_trimmed_wins)

    # Extra input context necessary to make up for windows lost on the right due to output trimming
    extra_right_context = max_(0, n_trimmed_wins * s_i - p_r)

    # Number of input elements that are needed as context
    in_context_size = max_(0, (windows_context_size - 1) * s_i + in_offset + extra_right_context)

    return s_i, s_o, in_offset, out_offset, in_context_size
