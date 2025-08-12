import math
from typing import Iterator, Tuple


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

    # TODO: test this function with a bunch of edge cases
    def get_min_input_size(self) -> int:
        """
        Returns the minimum input size necessary to have any output element (i.e. length>0). The returned value is
        always at least one.
        """
        out_needed = 1 + self.out_trim * 2
        num_wins_needed = int(math.ceil(max(0, out_needed - self.kernel_size_out) / self.stride_out)) + 1
        return self.get_min_input_size_for_num_wins(num_wins_needed)

    def get_min_input_size_for_num_wins(self, num_wins: int) -> int:
        """
        Returns the minimum input size necessary to have a given number of output windows.
        """
        non_padded_min_input_size = (num_wins - 1) * self.stride_in + self.kernel_size_in
        return max(1, non_padded_min_input_size - self.right_pad - self.left_pad)

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

    def __eq__(self, other):
        if not isinstance(other, SlidingWindowParams):
            return False
        return (
            self.kernel_size_in == other.kernel_size_in
            and self.kernel_size_out == other.kernel_size_out
            and self.stride_in == other.stride_in
            and self.left_pad == other.left_pad
            and self.right_pad == other.right_pad
            and self.stride_out == other.stride_out
            and self.out_trim == other.out_trim
        )

    def __hash__(self):
        return hash(
            (
                self.kernel_size_in,
                self.kernel_size_out,
                self.stride_in,
                self.left_pad,
                self.right_pad,
                self.stride_out,
                self.out_trim,
            )
        )

    def __repr__(self):
        return (
            "SlidingWindowParams(\n"
            + f"    kernel_size_in={self.kernel_size_in}, stride_in={self.stride_in},\n"
            + f"    left_pad={self.left_pad}, right_pad={self.right_pad},\n"
            + f"    kernel_size_out={self.kernel_size_out}, stride_out={self.stride_out},\n"
            + f"    out_trim={self.out_trim},\n"
            + ")"
        )
