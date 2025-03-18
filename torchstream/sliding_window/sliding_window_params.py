import math
from dataclasses import dataclass


@dataclass
class SlidingWindowParams:
    """
    This class represents the parameters of a sliding window transform (e.g. a conv1d layer).
    """

    # The kernel size of the input. For dilated (à trous) convolutions, this is the span of the entire kernel.
    kernel_size_in: int
    stride_in: int
    # The kernel size of the output. It is 1 for normal convolutions, but can be larger for transposed convolutions.
    kernel_size_out: int
    stride_out: int
    # The static number of elements to pad on the left side of the input.
    left_pad: int
    # This value represents the number of extra windows that are produced by the addition of the left and the right
    # padding combined. It is a proxy for right padding, because right padding is not necessarily constant, but this
    # value typically is. A few examples:
    #   - For a convolution with no padding, alpha = 0.
    #   - For a convolution with stride=1 and "same" padding (input size = output size), alpha = kernel_size_in - 1.
    alpha: int

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

    def __repr__(self):
        return (
            "SlidingWindowParams(\n"
            + f"    left_pad: {self.left_pad}, alpha={self.alpha},\n"
            + f"    kernel_size_in: {self.kernel_size_in}, stride_in: {self.stride_in},\n"
            + f"    kernel_size_out: {self.kernel_size_out}, stride_out: {self.stride_out},\n"
            + ")"
        )
