from dataclasses import dataclass, field
from typing import Tuple

import numpy as np


@dataclass
class SlidingWindowParams:
    """
    This class represents the parameters of a sliding window transform (e.g. a convolution, a moving average, ...).
    """

    # TODO: does default_factory actually cast?
    # The kernel size of the input. For dilated (à trous) convolutions, this is the span of the entire kernel.
    kernel_size_in: int = field(default_factory=int)
    stride_in: int = field(default_factory=int)
    # The static number of elements to pad on the left side of the input.
    left_pad: int = field(default_factory=int)
    # This value represents the number of extra windows that are produced as a result of the left and right padding
    # combined. It is a proxy for right padding, because right padding is not necessarily constant, but this
    # value typically is. A few examples:
    #   - For a conv1d layer with no padding, alpha = 0.
    #   - For a conv1d layer with stride=1 and "same" padding (input size = output size), alpha = kernel_size_in - 1.
    # FIXME!! change doc repo wide
    right_pad: int = field(default_factory=int)
    # The kernel size of the output. It is 1 for normal convolutions, but can be larger for transposed convolutions.
    kernel_size_out: int = field(default=1)
    stride_out: int = field(default=1)

    def __post_init__(self):
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

    def get_num_windows(self, input_size: int) -> int:
        """
        Returns the number of sliding windows (!= output size) that would be applied for an input of the given size.

        :param input_size: The size of the input tensor, without the sliding window padding applied.
        """
        if input_size <= 0:
            return 0
        return max(0, (self.left_pad + input_size + self.right_pad - self.kernel_size_in) // self.stride_in + 1)

    def get_effective_padding(self, input_size: int) -> Tuple[int, int]:
        """
        Returns the padding that would be applied to the input tensor before applying the sliding window transform.
        Because sliding windows with input stride > 1 might not line up exactly with the end of the padded input, the
        right padding for a given input size might be effectively less than <self.right_pad>. There are also cases
        where inputs on the right go unused, and therefore the effective right padding returned will be negative.

        :param input_size: The size of the input tensor, without padding added.
        """
        num_wins = self.get_num_windows(input_size)
        if num_wins == 0:
            # TODO: check if this really makes sense
            return (0, 0)

        padded_input_size = (num_wins - 1) * self.stride_in + self.kernel_size_in
        right_pad = padded_input_size - input_size - self.left_pad
        assert -self.stride_in < right_pad, "Internal error: trimming on the right should be smaller than the stride"
        assert right_pad <= self.kernel_size_in, "Internal error: padding on either side should not exceed kernel size"

        return (self.left_pad, right_pad)

    def get_output_size(self, input_size: int) -> int:
        """
        Returns the size of the output tensor after applying the sliding window transform.

        :param input_size: The size of the input tensor, without padding added.
        """
        num_wins = self.get_num_windows(input_size)
        if num_wins == 0:
            return 0
        return (num_wins - 1) * self.stride_out + self.kernel_size_out

    def get_min_input_size(self) -> int:
        """
        Returns the minimum input size necessary to have any output.
        This class considers that running a sliding window on an empty input is pointless and therefore the returned
        value is always at least one.
        """
        return max(1, self.kernel_size_in - self.right_pad - self.left_pad)

    def get_inverse_map(self, input_size: int) -> np.ndarray:
        # TODO! doc
        out_size = self.get_output_size(input_size)
        if not out_size:
            return np.zeros((0, 2), dtype=np.int64)
        left_pad, right_pad = self.get_effective_padding(input_size)
        num_wins = self.get_num_windows(input_size)

        padded_in_size = left_pad + input_size + right_pad

        out = np.zeros((out_size, 2), dtype=np.int64)
        out[:, 0] = padded_in_size
        out[:, 1] = -left_pad
        for i in range(num_wins):
            start_in_idx = i * self.stride_in - left_pad
            end_in_idx = i * self.stride_in + self.kernel_size_in - left_pad
            out_sli = slice(i * self.stride_out, i * self.stride_out + self.kernel_size_out)
            out[out_sli, 0] = np.minimum(out[out_sli, 0], start_in_idx)
            out[out_sli, 1] = np.maximum(out[out_sli, 1], end_in_idx)

        return out

    def __repr__(self):
        return (
            "SlidingWindowParams(\n"
            + f"    kernel_size_in={self.kernel_size_in}, stride_in={self.stride_in},\n"
            + f"    left_pad={self.left_pad}, right_pad={self.right_pad},\n"
            + f"    kernel_size_out={self.kernel_size_out}, stride_out={self.stride_out},\n"
            + ")"
        )
