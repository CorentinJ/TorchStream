import math
from typing import Iterator, Tuple, overload

from z3 import If, Int

from torchstream.sliding_window.z3_utils import IntLike, z3_ceil_div, z3_divmod, z3_floor_div, z3_max, z3_min


class SlidingWindowParams:
    """
    This class represents the parameters of a sliding window transform (e.g. a convolution, a moving average, ...).
    """

    def __init__(
        self,
        kernel_size_in: int = 1,
        stride_in: int = 1,
        left_pad: int = 0,
        right_pad: int = 0,
        kernel_size_out: int = 1,
        stride_out: int = 1,
        left_out_trim: int = 0,
        right_out_trim: int = 0,
        min_input_size: int | None = None,
    ):
        """
        :param kernel_size_in: The kernel size of the input. For dilated (à trous) convolutions, this is the span of
        the entire kernel.
        :param left_pad: The static number of elements to pad on the left side of the input.
        :param right_pad: The maximum number of elements to pad on the right side of the input. Due to windows not
        necessarily lining up with the input size with stride_in > 1, the effective right padding might be less than
        this value.
        :param kernel_size_out: The kernel size of the output. It is 1 for normal convolutions, but can be larger for
        transposed convolutions.
        :param left_out_trim: The number of elements to trim from the left of the output. It is uncommon to trim the
        output in practice, typically it's for getting rid of non-fully overlapping windows of the output when the
        output kernel size is larger than 1. Transposed convolutions expose this parameter through the "padding"
        parameter for example.
        :param right_out_trim: The number of elements to trim from the right of the output. Transposed convolutions
        allow different trimming values with their "output_padding" parameter.
        :param min_input_size: If provided, allows setting a higher bound on the minimum input size necessary to have
        any output element. This is useful for transforms that have a hard minimum input size requirement, such as
        reflect padding.
        """
        self.kernel_size_in = int(kernel_size_in)
        self.kernel_size_out = int(kernel_size_out)
        self.stride_in = int(stride_in)
        self.left_pad = int(left_pad)
        self.right_pad = int(right_pad)
        self.stride_out = int(stride_out)
        self.left_out_trim = int(left_out_trim)
        self.right_out_trim = int(right_out_trim)

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
        if self.left_out_trim < 0 or self.left_out_trim >= self.kernel_size_out:
            raise ValueError("left_out_trim must be at least 0 and at most kernel_size_out - 1.")
        if self.right_out_trim < 0 or self.right_out_trim >= self.kernel_size_out:
            raise ValueError("right_out_trim must be at least 0 and at most kernel_size_out - 1.")

        native_min_input_size = self.native_min_input_size
        if min_input_size is not None and min_input_size < native_min_input_size:
            raise ValueError(
                f"min_input_size must be at least {native_min_input_size}, the minimum input size "
                f"implied by the other parameters."
            )
        self.min_input_size = max(min_input_size or 1, native_min_input_size)

    @property
    def canonicalized_in_out_shape_params(self) -> Tuple[int, int, int, int]:
        return get_canonicalized_in_out_shape_params(self)

    @property
    def output_delay_bounds(self) -> Tuple[int, int]:
        return get_output_delay_bounds(self)

    @property
    def output_delays(self) -> Tuple[int, ...]:
        return get_all_output_delays(self)

    @property
    def streaming_context_size(self) -> int:
        return get_streaming_context_size(self)

    # TODO: test this function with a bunch of edge cases
    @property
    def native_min_input_size(self) -> int:
        """
        Returns the minimum input size necessary to have any output element (i.e. length>0). The returned value is
        always at least one.
        """
        out_needed = 1 + self.left_out_trim + self.right_out_trim
        num_wins_needed = int(math.ceil(max(0, out_needed - self.kernel_size_out) / self.stride_out)) + 1
        non_padded_min_input_size = (num_wins_needed - 1) * self.stride_in + self.kernel_size_in
        return max(1, non_padded_min_input_size - self.left_pad - self.right_pad)

    def get_min_input_size_for_num_wins(self, num_wins: int) -> int:
        """
        Returns the minimum input size necessary to have a given number of output windows.
        """
        non_padded_min_input_size = (num_wins - 1) * self.stride_in + self.kernel_size_in
        return max(self.min_input_size, non_padded_min_input_size - self.left_pad - self.right_pad)

    def get_min_input_size_for_out_size(self, out_size: int) -> int:
        """
        Returns the minimum input size necessary to have a given output size.
        """
        pre_trim_out_size = out_size + self.left_out_trim + self.right_out_trim
        num_wins_needed = int(math.ceil(max(0, pre_trim_out_size - self.kernel_size_out) / self.stride_out)) + 1
        return self.get_min_input_size_for_num_wins(num_wins_needed)

    # TODO! refactor, terrible name & mechanics
    def get_metrics_for_input(self, in_len: int) -> Tuple[Tuple[int, int], int, int]:
        """
        Computes the padding, number of windows and output length for an input to the transform with a given length.

        :param in_len: The length of the input tensor, without the sliding window padding applied.
        :return:
            - (left_pad, right_pad): A tuple of integers of the padding that is applied to the input tensor before
            applying the sliding window transform. Because sliding windows with input stride > 1 might not line up
            exactly with the end of the padded input, the right padding for a given input length might be effectively
            less than <self.right_pad>. There are also cases where inputs on the right go unused, and therefore the
            effective right padding returned will be negative. This is to ensure that the padded input always lines
            up with the last window.
            - num_wins: The number of windows that are computed for the given input length.
            - out_len: The length of the output tensor.
        """
        # Number of windows
        if in_len < self.min_input_size:
            num_wins = 0
        else:
            num_wins = (self.left_pad + in_len + self.right_pad - self.kernel_size_in) // self.stride_in + 1
            assert num_wins > 0

        # Padding
        if num_wins == 0:
            # TODO: check if this really makes sense
            #   -> Maybe return negative right padding??
            padding = (0, 0)
        else:
            padded_input_size = (num_wins - 1) * self.stride_in + self.kernel_size_in
            right_pad = padded_input_size - in_len - self.left_pad
            assert -self.stride_in < right_pad, (
                "Internal error: trimming on the right should be smaller than the stride"
            )
            assert right_pad <= self.kernel_size_in, (
                "Internal error: padding on either side should not exceed kernel size"
            )
            padding = (self.left_pad, right_pad)

        if num_wins == 0:
            out_len = 0
        else:
            out_len = (num_wins - 1) * self.stride_out + self.kernel_size_out - self.left_out_trim - self.right_out_trim
            assert out_len > 0

        return padding, num_wins, out_len

    def iter_kernel_map(
        self, *, in_len: int | None = None, num_wins: int | None = None
    ) -> Iterator[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """
        Iterates over the regions of input and output mapped by the sliding window transform.

        Note:
        - Both input and output windows may overlap.
        - Input windows will have negative bounds when overlapping with the left padding, and bounds beyond the input
        size when overlapping with the right padding.
        - Similarly, output windows will have negative bounds if they are to be trimmed on the left, and bounds beyond
        the output size when trimmed on the right.

        :param in_len: Length of the input tensor. Mutex with num_wins.
        :param num_wins: The number of windows to iterate over. Mutex with in_len, if neither are provided, iterates
        indefinitely.
        """
        if in_len is not None:
            if num_wins is not None:
                raise ValueError("Only one of in_len and num_wins should be provided.")
            _, num_wins, _ = self.get_metrics_for_input(in_len)
        elif num_wins is None:
            num_wins = int(1e10)
        else:
            num_wins = int(num_wins)

        for i in range(num_wins):
            yield (
                (
                    i * self.stride_in - self.left_pad,
                    i * self.stride_in + self.kernel_size_in - self.left_pad,
                ),
                (
                    i * self.stride_out - self.left_out_trim,
                    i * self.stride_out + self.kernel_size_out - self.left_out_trim,
                ),
            )

    def iter_bounded_kernel_map(
        self, in_len: int, bound_input: bool = True, bound_output: bool = True
    ) -> Iterator[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """
        Wrapper around get_kernel_map() that bounds the input and output windows between 0 and their respective sizes.
        """
        _, _, out_len = self.get_metrics_for_input(in_len)

        for (in_start, in_stop), (out_start, out_stop) in self.iter_kernel_map(in_len=in_len):
            if bound_input:
                in_start = min(max(in_start, 0), in_len)
                in_stop = min(max(in_stop, 0), in_len)
            if bound_output:
                out_start = min(max(out_start, 0), out_len)
                out_stop = min(max(out_stop, 0), out_len)

            yield (in_start, in_stop), (out_start, out_stop)

    def get_inverse_kernel_map(self, in_len: int):
        # TODO: doc
        _, num_wins, out_len = self.get_metrics_for_input(in_len)

        # Compute the overlapping output windows with the following output format:
        #   [(out_start, out_end, [(win_idx, kernel_out_start, kernel_out_end), ...]), ...]
        # TODO: this is O(n^2) when it can be done in O(n)
        win_out_map = [
            (win_idx, out_start, out_stop)
            for win_idx, (_, (out_start, out_stop)) in enumerate(self.iter_kernel_map(num_wins=num_wins))
        ]
        transition_points = sorted(
            set([max(min(pt, out_len), 0) for _, start, stop in win_out_map for pt in (start, stop)])
        )
        out_map = [
            (
                out_start,
                out_stop,
                [
                    (win_idx, max(0, out_start - win_out_start), min(win_out_stop, out_stop) - win_out_start)
                    for win_idx, win_out_start, win_out_stop in win_out_map
                    if not (win_out_stop <= out_start or win_out_start >= out_stop)
                ],
            )
            for out_start, out_stop in zip(transition_points[:-1], transition_points[1:])
        ]

        return out_map

    def as_tuple(self, with_min_in_size: bool = True) -> Tuple[int, ...]:
        return (
            self.kernel_size_in,
            self.stride_in,
            self.left_pad,
            self.right_pad,
            self.kernel_size_out,
            self.stride_out,
            self.left_out_trim,
            self.right_out_trim,
        ) + ((self.min_input_size,) if with_min_in_size else ())

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
            + f"left_out_trim={self.left_out_trim}, right_out_trim={self.right_out_trim},\n"
            + mis_str
            + ")"
        )


def _get_sli_args(args):
    if len(args) == 1 and isinstance(args[0], SlidingWindowParams):
        p = args[0]
        return p.as_tuple(with_min_in_size=False)
    elif len(args) == 8:
        return args
    else:
        raise TypeError()


@overload
def get_canonicalized_in_out_shape_params(
    sli_params: SlidingWindowParams,
) -> Tuple[int, int, int, int]: ...
@overload
def get_canonicalized_in_out_shape_params(
    k_i: IntLike, s_i: IntLike, p_l: IntLike, p_r: IntLike, k_o: IntLike, s_o: IntLike, t_l: IntLike, t_r: IntLike
) -> Tuple[IntLike, IntLike, IntLike, IntLike]: ...
def get_canonicalized_in_out_shape_params(*args) -> Tuple[IntLike, IntLike, IntLike, IntLike]:
    k_i, s_i, p_l, p_r, k_o, s_o, t_l, t_r = _get_sli_args(args)

    in_size_bias = p_l + p_r - k_i
    out_size_bias = k_o - t_l - t_r

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
    t_l: IntLike,
    t_r: IntLike,
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
    (k_i, s_i, p_l, p_r, k_o, s_o, t_l, t_r), input_size = _get_sli_args(args[:-1]), args[-1]

    if as_phase:
        if isinstance(input_size, int) and isinstance(s_i, int):
            assert 0 <= input_size < s_i, "When using phase, input_size must be in [0, stride_in["
        phase = input_size
    else:
        phase = (p_l + input_size - k_i) % s_i

    n_right_pad_corrupted_wins = z3_floor_div(phase + p_r, s_i)
    output_delay_pre_trim = k_o + (n_right_pad_corrupted_wins - 1) * s_o
    # FIXME!! double check tr vs tl
    output_delay = z3_max(0, output_delay_pre_trim - t_r)

    return output_delay


@overload
def get_output_delay_bounds(sli_params: SlidingWindowParams) -> Tuple[int, int]: ...
@overload
def get_output_delay_bounds(
    k_i: IntLike, s_i: IntLike, p_l: IntLike, p_r: IntLike, k_o: IntLike, s_o: IntLike, t_l: IntLike, t_r: IntLike
) -> Tuple[IntLike, IntLike]: ...
def get_output_delay_bounds(*args) -> Tuple[IntLike, IntLike]:
    # TODO: doc
    k_i, s_i, p_l, p_r, k_o, s_o, t_l, t_r = _get_sli_args(args)
    return (
        get_output_delay(k_i, s_i, p_l, p_r, k_o, s_o, t_l, t_r, 0, as_phase=True),
        get_output_delay(k_i, s_i, p_l, p_r, k_o, s_o, t_l, t_r, s_i - 1, as_phase=True),
    )


@overload
def get_all_output_delays(sli_params: SlidingWindowParams) -> Tuple[int, ...]: ...
@overload
def get_all_output_delays(
    k_i: IntLike, s_i: IntLike, p_l: IntLike, p_r: IntLike, k_o: IntLike, s_o: IntLike, t_l: IntLike, t_r: IntLike
) -> Tuple[IntLike, ...]: ...
def get_all_output_delays(*args) -> Tuple[IntLike, ...]:
    # TODO: doc
    k_i, s_i, p_l, p_r, k_o, s_o, t_l, t_r = _get_sli_args(args)
    # NOTE: can be computed more efficiently for very large strides if necessary
    return tuple(get_output_delay(k_i, s_i, p_l, p_r, k_o, s_o, t_l, t_r, phase, as_phase=True) for phase in range(s_i))


@overload
def get_streaming_context_size(
    sli_params: SlidingWindowParams,
) -> int: ...
@overload
def get_streaming_context_size(
    k_i: IntLike, s_i: IntLike, p_l: IntLike, p_r: IntLike, k_o: IntLike, s_o: IntLike, t_l: IntLike, t_r: IntLike
) -> IntLike: ...
def get_streaming_context_size(*args) -> IntLike:
    """
    Get the input context size necessary for streaming a transform with given sliding window parameters.

    When streaming a transform, we continuously discard seen input in order to limit the compute cost of the transform.
    However, there is a certain minimum number of elements on the right that need not to be discarded in order for the
    output to be equivalent from its non-streamed version. This value is the context size and we can derive it from
    the sliding window parameters.
    """
    k_i, s_i, p_l, p_r, k_o, s_o, t_l, t_r = _get_sli_args(args)

    in_delay = p_l + p_r - k_i
    in_delay_n_wins, in_delay_remainder = z3_divmod(in_delay, s_i)

    # TODO!! verify
    last_left_incomplete_out_idx = z3_ceil_div(p_l, s_i) * s_o + (k_o - 1) - t_l

    def ctx_for_remainder(remainder: IntLike) -> IntLike:
        out_delay = get_output_delay(k_i, s_i, p_l, p_r, k_o, s_o, t_l, t_r, remainder)
        effective_out_core = k_o - t_l - t_r - out_delay

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


def in_out_rel_repr(s_i: int, s_o: int, isb: int, osb: int) -> str:
    # I didn't want to use sympy for this...
    out_str = "x"
    if isb > 0:
        out_str = f"(x + {isb})"
    elif isb < 0:
        out_str = f"(x - {-isb})"
    if s_i > 1:
        out_str = f"({out_str} // {s_i})"
    if s_o > 1:
        out_str = f"{out_str} * {s_o}"
    if osb > 0:
        out_str = f"{out_str} + {osb}"
    elif osb < 0:
        out_str = f"{out_str} - {-osb}"

    return f"y = {out_str}"
