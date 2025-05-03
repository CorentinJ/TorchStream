from dataclasses import dataclass, field
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class SlidingWindowParams:
    """
    This class represents the parameters of a sliding window transform (e.g. a convolution, a moving average, ...).
    """

    # TODO: does default_factory actually cast?
    # The kernel size of the input. For dilated (à trous) convolutions, this is the span of the entire kernel.
    # TODO: clarify these params are not only for convs
    kernel_size_in: int = field(default=1)
    stride_in: int = field(default=1)
    # The static number of elements to pad on the left side of the input.
    left_pad: int = field(default=0)
    # This value represents the number of extra windows that are produced as a result of the left and right padding
    # combined. It is a proxy for right padding, because right padding is not necessarily constant, but this
    # value typically is. A few examples:
    #   - For a conv1d layer with no padding, alpha = 0.
    #   - For a conv1d layer with stride=1 and "same" padding (input size = output size), alpha = kernel_size_in - 1.
    # FIXME!! change doc repo wide
    right_pad: int = field(default=0)
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

    def get_metrics_for_input(self, input_size: int) -> Tuple[Tuple[int, int], int, int]:
        """
        Computes the padding, number of windows and output size for an input to the transform with a given size.

        :param input_size: The size of the input tensor, without the sliding window padding applied.
        :return:
            - (left_pad, right_pad): A tuple of integers of the padding that is applied to the input tensor before
            applying the sliding window transform. Because sliding windows with input stride > 1 might not line up
            exactly with the end of the padded input, the right padding for a given input size might be effectively
            less than <self.right_pad>. There are also cases where inputs on the right go unused, and therefore the
            effective right padding returned will be negative. This is to ensure that the padded input always lines
            up with the last window.
            - num_wins: The number of windows that are computed for the given input size.
            - out_size: The size of the output tensor.
        """
        # Number of windows
        if input_size <= 0:
            num_wins = 0
        else:
            num_wins = max(0, (self.left_pad + input_size + self.right_pad - self.kernel_size_in) // self.stride_in + 1)

        # Padding
        if num_wins == 0:
            # TODO: check if this really makes sense
            #   -> Maybe return negative right padding??
            padding = (0, 0)
        else:
            padded_input_size = (num_wins - 1) * self.stride_in + self.kernel_size_in
            right_pad = padded_input_size - input_size - self.left_pad
            assert -self.stride_in < right_pad, (
                "Internal error: trimming on the right should be smaller than the stride"
            )
            assert right_pad <= self.kernel_size_in, (
                "Internal error: padding on either side should not exceed kernel size"
            )
            padding = (self.left_pad, right_pad)

        if num_wins == 0:
            output_size = 0
        else:
            output_size = (num_wins - 1) * self.stride_out + self.kernel_size_out

        return padding, num_wins, output_size

    def get_min_input_size(self) -> int:
        """
        Returns the minimum input size necessary to have any output.
        This class considers that running a sliding window on an empty input is pointless and therefore the returned
        value is always at least one.
        """
        return max(1, self.kernel_size_in - self.right_pad - self.left_pad)

    def get_inverse_map(self, input_size: int) -> np.ndarray:
        (left_pad, right_pad), num_wins, out_size = self.params.get_metrics_for_input(input_size)
        if not out_size:
            return np.zeros((0, 2), dtype=np.int64)

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
