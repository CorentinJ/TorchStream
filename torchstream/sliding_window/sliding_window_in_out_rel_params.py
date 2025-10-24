import math
from typing import Tuple

from z3 import Int

from torchstream.transforms.z3_utils import IntLike, z3_floor_div, z3_max


class SlidingWindowInOutRelParams:
    """
    Parameters that determine the size relation between an input and output sequence for a sliding window transform.

    The relation is defined as:
        out_size = ((in_size + input_size_bias) // stride_in) * stride_out + output_size_bias
    for in_size >= min_input_size (else out_size = 0).
    """

    def __init__(
        self,
        stride_in: int,
        input_size_bias: int,
        stride_out: int,
        output_size_bias: int,
        min_input_size: int | None = None,
    ):
        if stride_in < 1:
            raise ValueError("stride_in must be at least 1")
        if stride_out < 1:
            raise ValueError("stride_out must be at least 1")

        self.stride_in = int(stride_in)
        self.input_size_bias = int(input_size_bias)
        self.stride_out = int(stride_out)
        self.output_size_bias = int(output_size_bias)

        self.input_size_bias_canon, self.output_size_bias_canon = get_canonicalized_in_out_size_biases(
            self.stride_in,
            self.input_size_bias,
            self.stride_out,
            self.output_size_bias,
        )

        self.native_min_input_size = get_native_min_in_size(
            self.stride_in,
            self.stride_out,
            self.input_size_bias,
            self.output_size_bias,
        )

        if min_input_size is not None and min_input_size < self.native_min_input_size:
            raise ValueError(
                f"min_input_size must be at least {self.native_min_input_size}, the minimum input size "
                f"implied by the other parameters."
            )
        self.min_input_size = max(min_input_size or 1, self.native_min_input_size)

    @property
    def in_out_rel_params_tuple(self) -> Tuple[int, int, int, int, int]:
        return self.stride_in, self.input_size_bias, self.stride_out, self.output_size_bias, self.min_input_size

    @property
    def canon_in_out_rel_params_tuple(self) -> Tuple[int, int, int, int, int]:
        return (
            self.stride_in,
            self.input_size_bias_canon,
            self.stride_out,
            self.output_size_bias_canon,
            self.min_input_size,
        )

    def get_out_size(self, in_size: int) -> int:
        """
        Returns the output size for a given input size.
        """
        num_wins = self.get_num_wins(in_size)
        if num_wins == 0:
            return 0
        else:
            return (num_wins - 1) * self.stride_out + self.output_size_bias

    def get_min_input_size_for_out_size(self, out_size: int) -> int:
        """
        Returns the minimum input size necessary to have a given output size.
        """
        num_wins_needed = int(math.ceil(max(0, out_size - self.output_size_bias) / self.stride_out)) + 1
        return self.get_min_input_size_for_num_wins(num_wins_needed)

    def get_num_wins(self, in_size: int) -> int:
        """
        Returns the number of windows computed for a given input size.
        NOTE: The function returns 0 if the input is less than the minimum input size. That includes the case where
        there are >0 windows computed but the output size is 0 due to trimming.
        """
        if in_size < self.min_input_size:
            return 0
        else:
            return (in_size + self.input_size_bias) // self.stride_in + 1

    def get_min_input_size_for_num_wins(self, num_wins: int) -> int:
        """
        Returns the minimum input size necessary to compute at least the given number of windows.
        """
        in_size_necessary = (num_wins - 1) * self.stride_in - self.input_size_bias
        return max(self.min_input_size, in_size_necessary)

    def __eq__(self, other):
        if not isinstance(other, SlidingWindowInOutRelParams):
            return False
        return self.in_out_rel_params_tuple == other.in_out_rel_params_tuple

    def __hash__(self):
        return hash(self.as_tuple())

    def __repr__(self):
        # I didn't want to use sympy for this...
        out_str = "x"
        if self.input_size_bias > 0:
            out_str = f"(x + {self.input_size_bias})"
        else:
            out_str = f"(x - {-self.input_size_bias})"
        if self.stride_in > 1:
            out_str = f"({out_str} // {self.stride_in})"
        if self.stride_out > 1:
            out_str = f"{out_str} * {self.stride_out}"
        if self.output_size_bias > 0:
            out_str = f"{out_str} + {self.output_size_bias}"
        elif self.output_size_bias < 0:
            out_str = f"{out_str} - {-self.output_size_bias}"

        if self.min_input_size > 1:
            out_str += f", for x >= {self.min_input_size}"

        return f"InOutRelParams: y = {out_str}"


def get_canonicalized_in_out_size_biases(
    s_i: IntLike, isb: IntLike, s_o: IntLike, osb: IntLike
) -> Tuple[IntLike, IntLike]:
    """
    Returns the canonicalized in/out size biases. With the same in/out strides, these parameters express the same
    in->out size relation in a unique manner.
    """
    if isinstance(s_i, int) and isinstance(isb, int):
        quotient_bias, isbc = divmod(isb, s_i)
    else:
        quotient_bias = Int("quotient_bias")
        isbc = isb - quotient_bias * s_i

    osbc = osb + quotient_bias * s_o

    return isbc, osbc


def get_native_min_in_size(s_i: IntLike, s_o: IntLike, isb: IntLike, osb: IntLike) -> IntLike:
    """
    Returns the minimum input size necessary to have any output element. This is the "native" minimum input size
    because a relation may impose a higher minimum input size (e.g. reflect padding does this).
    """
    return z3_max((z3_floor_div(-osb, s_o) + 1) * s_i - isb, 1)
