import math
from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class SlidingWindowParams:
    """
    This class represents the parameters of a sliding window transform (e.g. a convolution, a moving average, ...).
    """

    # The kernel size of the input. For dilated (à trous) convolutions, this is the span of the entire kernel.
    kernel_size_in: int = field(default_factory=int)
    stride_in: int = field(default_factory=int)
    # The kernel size of the output. It is 1 for normal convolutions, but can be larger for transposed convolutions.
    kernel_size_out: int = field(default_factory=int)
    stride_out: int = field(default_factory=int)
    # The static number of elements to pad on the left side of the input.
    left_pad: int = field(default_factory=int)
    # This value represents the number of extra windows that are produced as a result of the left and right padding
    # combined. It is a proxy for right padding, because right padding is not necessarily constant, but this
    # value typically is. A few examples:
    #   - For a conv1d layer with no padding, alpha = 0.
    #   - For a conv1d layer with stride=1 and "same" padding (input size = output size), alpha = kernel_size_in - 1.
    alpha: int = field(default_factory=int)

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
        if self.alpha < 0 or self.alpha > 2 * (self.kernel_size_in - 1):
            raise ValueError("alpha must be at least 0 and at most 2 * (kernel_size_in - 1).")
        if math.ceil(self.left_pad / self.stride_in) > self.alpha:
            raise ValueError("the left padding is excessive given the alpha value.")

    def get_num_windows(self, input_size: int) -> int:
        """
        Returns the number of sliding windows (!= output size) that would be applied for an input of the given size.

        :param input_size: The size of the input tensor, without the sliding window padding applied.
        """
        return max(0, (input_size - self.kernel_size_in) // self.stride_in + self.alpha + 1)

    def get_padding(self, input_size: int) -> Tuple[int, int]:
        """
        Returns the padding that would be applied to the input tensor before applying the sliding window transform.

        Note that the right padding might be negative, implying that the input tensor gets trimmed on the right
        instead of being padded.

        :param input_size: The size of the input tensor, without the sliding window padding applied.
        """
        num_wins = self.get_num_windows(input_size)
        padded_input_size = (num_wins - 1) * self.stride_in + self.kernel_size_in
        right_pad = padded_input_size - input_size - self.left_pad
        if self.left_pad > 0:
            assert right_pad > 0, "Internal error: right padding should be positive if left padding is positive."
        return (self.left_pad, right_pad)

    def get_output_size(self, input_size: int) -> int:
        """
        Returns the size of the output tensor after applying the sliding window transform.

        :param input_size: The size of the input tensor, without the sliding window padding applied.
        """
        num_wins = self.get_num_windows(input_size)
        if num_wins == 0:
            return 0
        return (num_wins - 1) * self.stride_out + self.kernel_size_out

    def __repr__(self):
        return (
            "SlidingWindowParams(\n"
            + f"    left_pad: {self.left_pad}, alpha={self.alpha},\n"
            + f"    kernel_size_in: {self.kernel_size_in}, stride_in: {self.stride_in},\n"
            + f"    kernel_size_out: {self.kernel_size_out}, stride_out: {self.stride_out},\n"
            + ")"
        )
