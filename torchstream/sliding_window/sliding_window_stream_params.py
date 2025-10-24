import math
from typing import Iterable, Iterator, Tuple, overload

from z3 import If, Int

from torchstream.sliding_window.sliding_window_in_out_rel_params import SlidingWindowInOutRelParams
from torchstream.transforms.z3_utils import IntLike, z3_ceil_div, z3_divmod, z3_floor_div, z3_max, z3_min


class SlidingWindowStreamParams:
    """ """

    def __init__(
        self,
        in_out_rel: SlidingWindowInOutRelParams,
        out_delays: Iterable[int],
        streaming_context_size: int,
    ):
        self.output_delays = [int(d) for d in out_delays]
        self.streaming_context_size = int(streaming_context_size)

        if self.kernel_size_in < 1:
            raise ValueError("kernel_size_in must be at least 1.")
        if self.stride_in < 1 or self.stride_in > self.kernel_size_in:
            raise ValueError("stride_in must be at least 1 and at most kernel_size_in.")
        if self.kernel_size_out < 1:
            raise ValueError("kernel_size_out must be at least 1.")
        if self.stride_out < 1 or self.stride_out > self.kernel_size_out:
            raise ValueError("stride_out must be at least 1 and at most kernel_size_out.")
        if self.left_pad < 0 or self.left_pad >= self.kernel_size_in:
            raise ValueError("left_pad must be at least 0 and at most kernel_size_in - 1.")
        if self.right_pad < 0 or self.right_pad >= self.kernel_size_in:
            raise ValueError("right_pad must be at least 0 and at most kernel_size_in - 1.")
        if self.out_trim < 0 or self.out_trim >= self.kernel_size_out:
            raise ValueError("out_trim must be at least 0 and at most kernel_size_out - 1.")

        native_min_input_size = self.native_min_input_size
        if min_input_size is not None and min_input_size < native_min_input_size:
            raise ValueError(
                f"min_input_size must be at least {native_min_input_size}, the minimum input size "
                f"implied by the other parameters."
            )
        self.min_input_size = max(min_input_size or 1, native_min_input_size)

    @property
    def canonicalized_in_out_size_params(self) -> Tuple[int, int, int, int, int]:
        # FIXME!
        return get_canonicalized_in_out_size_params(self) + (self.min_input_size,)

    @property
    def output_delay_bounds(self) -> Tuple[int, int]:
        return get_output_delay_bounds(self)

    @property
    def output_delays(self) -> Tuple[int, ...]:
        return get_all_output_delays(self)

    @property
    def streaming_context_size(self) -> int:
        return get_streaming_context_size(self)

    def __eq__(self, other):
        if not isinstance(other, SlidingWindowParams):
            return False
        return self.as_tuple() == other.as_tuple()

    def __hash__(self):
        return hash(self.as_tuple())

    def __repr__(self):
        mis_str = (
            "" if self.min_input_size == self.native_min_input_size else f"    min_input_size={self.min_input_size},\n"
        )
        return (
            "SlidingWindowParams(\n"
            + f"    kernel_size_in={self.kernel_size_in}, stride_in={self.stride_in}, "
            + f"left_pad={self.left_pad}, right_pad={self.right_pad},\n"
            + f"    kernel_size_out={self.kernel_size_out}, stride_out={self.stride_out}, "
            + f"out_trim={self.out_trim},\n"
            + mis_str
            + ")"
        )

@overload
def get_canonicalized_in_out_size_params(
    sli_params: SlidingWindowParams,
) -> Tuple[int, int, int, int]: ...
@overload
def get_canonicalized_in_out_size_params(
    k_i: IntLike, s_i: IntLike, p_l: IntLike, p_r: IntLike, k_o: IntLike, s_o: IntLike, t_o: IntLike
) -> Tuple[IntLike, IntLike, IntLike, IntLike]: ...
def get_canonicalized_in_out_size_params(*args) -> Tuple[IntLike, IntLike, IntLike, IntLike]:
    k_i, s_i, p_l, p_r, k_o, s_o, t_o = _get_sli_args(args)

    in_size_bias = p_l + p_r - k_i
    out_size_bias = k_o - 2 * t_o

    # Make the biases canonical so size relations are uniquely determined by a set of parameters
    if isinstance(s_i, int) and isinstance(in_size_bias, int):
        quotient_bias, in_size_bias_canon = divmod(in_size_bias, s_i)
    else:
        quotient_bias = Int("quotient_bias")
        in_size_bias_canon = in_size_bias - quotient_bias * s_i

    out_size_bias_canon = out_size_bias + quotient_bias * s_o

    return s_i, s_o, in_size_bias_canon, out_size_bias_canon


def get_canonicalized_min_in_size(s_i: IntLike, s_o: IntLike, isbc: IntLike, osbc: IntLike) -> IntLike:
    return z3_max((z3_floor_div(-osbc, s_o) + 1) * s_i - isbc, 1)


@overload
def get_output_delay(sli_params: SlidingWindowParams, input_size: int, as_phase=False) -> int: ...
@overload
def get_output_delay(
    k_i: IntLike,
    s_i: IntLike,
    p_l: IntLike,
    p_r: IntLike,
    k_o: IntLike,
    s_o: IntLike,
    t_o: IntLike,
    input_size: int,
    as_phase=False,
) -> IntLike: ...
def get_output_delay(*args, as_phase=False) -> IntLike:
    """
    Computes the streaming output delay for the sliding window parameters. Given an input sequence, the output delay
    is the number of elements at the end of its output sequence that will no longer be correct if more output is to be
    produced with new input elements, i.e. if we're doing streaming.

    Therefore when streaming, we keep outputs up to out_len - output_delay and discard the rest.

    The output delay is constant for parameters right padding=0, but with right padding>0 it can take two different
    values depending on the phase (i.e. on the input size).

    """
    (k_i, s_i, p_l, p_r, k_o, s_o, t_o), input_size = _get_sli_args(args[:-1]), args[-1]

    if as_phase:
        if isinstance(input_size, int) and isinstance(s_i, int):
            assert 0 <= input_size < s_i, "When using phase, input_size must be in [0, stride_in["
        phase = input_size
    else:
        phase = (p_l + input_size - k_i) % s_i

    n_right_pad_corrupted_wins = z3_floor_div(phase + p_r, s_i)
    output_delay_pre_trim = k_o + (n_right_pad_corrupted_wins - 1) * s_o
    output_delay = z3_max(0, output_delay_pre_trim - t_o)

    return output_delay


@overload
def get_output_delay_bounds(sli_params: SlidingWindowParams) -> Tuple[int, int]: ...
@overload
def get_output_delay_bounds(
    k_i: IntLike, s_i: IntLike, p_l: IntLike, p_r: IntLike, k_o: IntLike, s_o: IntLike, t_o: IntLike
) -> Tuple[IntLike, IntLike]: ...
def get_output_delay_bounds(*args) -> Tuple[IntLike, IntLike]:
    # TODO: doc
    k_i, s_i, p_l, p_r, k_o, s_o, t_o = _get_sli_args(args)
    return (
        get_output_delay(k_i, s_i, p_l, p_r, k_o, s_o, t_o, 0, as_phase=True),
        get_output_delay(k_i, s_i, p_l, p_r, k_o, s_o, t_o, s_i - 1, as_phase=True),
    )


@overload
def get_all_output_delays(sli_params: SlidingWindowParams) -> Tuple[int, ...]: ...
@overload
def get_all_output_delays(
    k_i: IntLike, s_i: IntLike, p_l: IntLike, p_r: IntLike, k_o: IntLike, s_o: IntLike, t_o: IntLike
) -> Tuple[IntLike, ...]: ...
def get_all_output_delays(*args) -> Tuple[IntLike, ...]:
    # TODO: doc
    k_i, s_i, p_l, p_r, k_o, s_o, t_o = _get_sli_args(args)
    # NOTE: can be computed more efficiently for very large strides if necessary
    return tuple(get_output_delay(k_i, s_i, p_l, p_r, k_o, s_o, t_o, phase, as_phase=True) for phase in range(s_i))


@overload
def get_streaming_context_size(
    sli_params: SlidingWindowParams,
) -> int: ...
@overload
def get_streaming_context_size(
    k_i: IntLike, s_i: IntLike, p_l: IntLike, p_r: IntLike, k_o: IntLike, s_o: IntLike, t_o: IntLike
) -> IntLike: ...
def get_streaming_context_size(*args) -> IntLike:
    """
    Get the input context size necessary for streaming a transform with given sliding window parameters.

    When streaming a transform, we continuously discard seen input in order to limit the compute cost of the transform.
    However, there is a certain minimum number of elements on the right that need not to be discarded in order for the
    output to be equivalent from its non-streamed version. This value is the context size and we can derive it from
    the sliding window parameters.
    """
    k_i, s_i, p_l, p_r, k_o, s_o, t_o = _get_sli_args(args)

    in_delay = p_l + p_r - k_i
    in_delay_n_wins, in_delay_remainder = z3_divmod(in_delay, s_i)

    last_left_incomplete_out_idx = z3_ceil_div(p_l, s_i) * s_o + (k_o - 1) - t_o

    def ctx_for_remainder(remainder: IntLike) -> IntLike:
        out_delay = get_output_delay(k_i, s_i, p_l, p_r, k_o, s_o, t_o, remainder)
        effective_out_core = k_o - 2 * t_o - out_delay

        if isinstance(in_delay_remainder, int) and isinstance(remainder, int):
            bias_carry = 1 if (in_delay_remainder + remainder) >= s_i else 0
        else:
            bias_carry = If(in_delay_remainder + remainder >= s_i, 1, 0)

        min_wins_vs_left_incomplete = z3_ceil_div(effective_out_core - last_left_incomplete_out_idx, s_o)
        min_wins_vs_core = z3_floor_div(effective_out_core, s_o)
        wins_to_keep = -in_delay_n_wins - bias_carry - z3_min(min_wins_vs_left_incomplete, min_wins_vs_core)
        return z3_max(0, (wins_to_keep - 1) * s_i + remainder + 1)

    r_best = (s_i - in_delay_remainder - 1) % s_i
    r_neighbor = (r_best + 1) % s_i
    r_delay = (k_i - p_l - 1) % s_i
    in_context_size = z3_max(
        z3_max(ctx_for_remainder(r_best), ctx_for_remainder(r_neighbor)), ctx_for_remainder(r_delay)
    )

    return in_context_size
