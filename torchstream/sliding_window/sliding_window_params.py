import math
from typing import Iterator, Tuple, Union, overload

from z3 import And, ArithRef, If, Int


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
        out_trim: int = 0,
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
        :param out_trim: The number of elements to trim from both sides of the output. It is rare to trim the output
        in practice, typically it's for getting rid of non-fully overlapping windows of the output when the output
        kernel size is larger than 1. Transposed convolutions expose this parameter through the "padding" parameter
        for example.
        NOTE: So far I haven't met a model that had different left/right values, output padding or trimming
        larger than the kernel size.
        """
        self.kernel_size_in = int(kernel_size_in)
        self.kernel_size_out = int(kernel_size_out)
        self.stride_in = int(stride_in)
        self.left_pad = int(left_pad)
        self.right_pad = int(right_pad)
        self.stride_out = int(stride_out)
        self.out_trim = int(out_trim)

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

    @property
    def canonicalized_in_out_size_params(self) -> Tuple[int, int, int, int]:
        return get_canonicalized_in_out_size_params(self)

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
    def min_input_size(self) -> int:
        """
        Returns the minimum input size necessary to have any output element (i.e. length>0). The returned value is
        always at least one.
        """
        out_needed = 1 + self.out_trim * 2
        num_wins_needed = int(math.ceil(max(0, out_needed - self.kernel_size_out) / self.stride_out)) + 1
        return self.get_min_input_size_for_num_wins(num_wins_needed)

    # TODO: get_min_input_size_for_out_size
    def get_min_input_size_for_num_wins(self, num_wins: int) -> int:
        """
        Returns the minimum input size necessary to have a given number of output windows.
        """
        non_padded_min_input_size = (num_wins - 1) * self.stride_in + self.kernel_size_in
        return max(1, non_padded_min_input_size - self.right_pad - self.left_pad)

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
        if in_len <= 0:
            num_wins = 0
        else:
            num_wins = max(0, (self.left_pad + in_len + self.right_pad - self.kernel_size_in) // self.stride_in + 1)

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
            out_len = max(0, (num_wins - 1) * self.stride_out + self.kernel_size_out - 2 * self.out_trim)

        return padding, num_wins, out_len

    def iter_kernel_map(self, num_wins: int | None = None) -> Iterator[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """
        Iterates over the regions of input and output mapped by the sliding window transform.

        Note:
        - Both input and output windows may overlap.
        - Input windows will have negative bounds when overlapping with the left padding, and bounds beyond the input
        size when overlapping with the right padding.
        - Similarly, output windows will have negative bounds if they are to be trimmed on the left, and bounds beyond
        the output size when trimmed on the right.

        :param num_wins: The number of windows to iterate over. If None, it will iterate without limit.
        """
        num_wins = num_wins if num_wins is not None else int(1e10)

        for i in range(num_wins):
            yield (
                (
                    i * self.stride_in - self.left_pad,
                    i * self.stride_in + self.kernel_size_in - self.left_pad,
                ),
                (
                    i * self.stride_out - self.out_trim,
                    i * self.stride_out + self.kernel_size_out - self.out_trim,
                ),
            )

    def iter_bounded_kernel_map(
        self, in_len: int, bound_input: bool = True, bound_output: bool = True
    ) -> Iterator[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """
        Wrapper around get_kernel_map() that bounds the input and output windows between 0 and their respective sizes.
        """
        _, num_wins, out_len = self.get_metrics_for_input(in_len)

        for (in_start, in_stop), (out_start, out_stop) in self.iter_kernel_map(num_wins):
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

        out_map = [[] for _ in range(out_len)]
        for win_idx, ((in_start, in_stop), (out_start, out_stop)) in enumerate(self.iter_kernel_map(num_wins)):
            for out_idx in range(max(0, out_start), min(out_len, out_stop)):
                out_map[out_idx].append((win_idx, in_start, in_stop, out_idx - out_start))

        return out_map

    def as_tuple(self) -> Tuple[int, int, int, int, int, int, int]:
        return (
            self.kernel_size_in,
            self.stride_in,
            self.left_pad,
            self.right_pad,
            self.kernel_size_out,
            self.stride_out,
            self.out_trim,
        )

    def __eq__(self, other):
        if not isinstance(other, SlidingWindowParams):
            return False
        return self.as_tuple() == other.as_tuple()

    def __hash__(self):
        return hash(self.as_tuple())

    def __repr__(self):
        return (
            "SlidingWindowParams(\n"
            + f"    kernel_size_in={self.kernel_size_in}, stride_in={self.stride_in},\n"
            + f"    left_pad={self.left_pad}, right_pad={self.right_pad},\n"
            + f"    kernel_size_out={self.kernel_size_out}, stride_out={self.stride_out},\n"
            + f"    out_trim={self.out_trim},\n"
            + ")"
        )


IntLike = Union[int, ArithRef]


# FIXME! more efficient expression of all constraints below
def _ceil_div(a, b) -> IntLike:
    """Ceiling division that works for both Python ints and z3 expressions."""
    if isinstance(a, int) and isinstance(b, int):
        return int(math.ceil(a / b))
    return If(a >= 0, (a + b - 1) / b, -((-a) / b))


def _floor_div(a, b) -> IntLike:
    """Floor division that works for both Python ints and z3 expressions."""
    if isinstance(a, int) and isinstance(b, int):
        return a // b
    return If(a >= 0, a / b, -((-a + b - 1) / b))


def _max(a: IntLike, b: IntLike) -> IntLike:
    """max() for both Python ints and z3 expressions."""
    if isinstance(a, int) and isinstance(b, int):
        return max(a, b)
    return If(a > b, a, b)


def _get_sli_args(args):
    if len(args) == 1 and isinstance(args[0], SlidingWindowParams):
        p = args[0]
        # TODO: order consistency with constructor & as_tuple()
        return (
            p.kernel_size_in,
            p.stride_in,
            p.left_pad,
            p.right_pad,
            p.kernel_size_out,
            p.stride_out,
            p.out_trim,
        )
    elif len(args) == 7:
        return args
    else:
        raise TypeError()


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
    # TODO doc
    (k_i, s_i, p_l, p_r, k_o, s_o, t_o), input_size = _get_sli_args(args[:-1]), args[-1]

    if as_phase:
        if isinstance(input_size, int) and isinstance(s_i, int):
            assert 0 <= input_size < s_i, "When using phase, input_size must be in [0, stride_in["
        phase = input_size
    else:
        phase = (p_l + input_size - k_i) % s_i

    n_right_pad_corrupted_wins = _floor_div(phase + p_r, s_i)
    output_delay_pre_trim = k_o + (n_right_pad_corrupted_wins - 1) * s_o
    output_delay = _max(0, output_delay_pre_trim - t_o)

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
    However, there is a certain minimum number of elements that need not to be discarded in order for the output to be
    equivalent from its non-streamed version. This value is the context size and we can derive it from the sliding
    window parameters.
    """
    k_i, s_i, p_l, p_r, k_o, s_o, t_o = _get_sli_args(args)

    # Number of windows that are wasted on the left solely due to padding. "Wasted" here means that we recompute
    # these windows on each step despite them being unnecessary, simply because the transform re-pads the input
    # every time. If it is possible to remove padding from the transform and manually pad the streamed input,
    # this waste of compute can be avoided.
    # Note that right padding wastes compute just as much, however it does not require any context to be stored.
    n_left_pad_wins_wasted = _ceil_div(p_l, s_i)

    # For a given output window, the number of other output windows that overlap it. Only >0 when the out stride
    # is smaller than the out kernel size.
    # Note that we need to buffer enough past context in order to have the overlapping windows necessary in
    # computing a given output. This induces redundant compute that could be avoided if the reduce operation on
    # overlapping windows (e.g. a vector sum) is known. TODO: test & implement if useful
    n_overlapping_out_wins = _ceil_div(k_o, s_o) - 1

    # FIXME! doc & names
    # Output trimming might trim away the content of output windows on the right. Depending on the output overlap,
    # we might need to compute additional windows every step to make up for that.
    # With any output trimming, we'll need an extra window if there no output window overlap at all
    if isinstance(s_o, int) and isinstance(k_o, int) and isinstance(t_o, int):
        n_extra_right_wins = 1 if t_o > (k_o - s_o) else 0
        boundary_window_needed = 1 if s_o == k_o and t_o > 0 else 0
    else:
        n_extra_right_wins = If(t_o > (k_o - s_o), 1, 0)
        boundary_window_needed = If(And(s_o == k_o, t_o > 0), 1, 0)

    # Convert the number of extra right windows into a number of input elements, offset by right padding that provides
    # extra right context for free
    extra_right_context = _max(0, (n_extra_right_wins + boundary_window_needed) * s_i - p_r)

    # Total number of input elements that are needed as context
    in_delay = k_i - s_i - p_l
    in_context_size = _max(0, (n_left_pad_wins_wasted + n_overlapping_out_wins) * s_i + in_delay + extra_right_context)

    return in_context_size
