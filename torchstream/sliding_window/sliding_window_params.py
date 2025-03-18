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
    # TODO: doc
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
        if self.left_pad < 0:
            raise ValueError("left_pad must be at least 0.")
        # TODO: constraint between left pad & alpha
        # if self.drop_right and self.left_pad != 0:
        # raise ValueError("When drop_right is True, left_pad must be 0.")

    def __repr__(self):
        return (
            "SlidingWindowParams(\n"
            + f"    left_pad: {self.left_pad}, alpha={self.alpha}\n"
            + f"    kernel_size_in: {self.kernel_size_in}, stride_in: {self.stride_in}\n"
            + f"    kernel_size_out: {self.kernel_size_out}, stride_out: {self.stride_out}\n"
            + ")"
        )
