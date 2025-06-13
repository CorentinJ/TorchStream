import math
from typing import Callable, Tuple, Union

import numpy as np
from torch import Tensor

from torchstream.sequence.seq_spec import SeqSpec
from torchstream.sequence.sequence import Sequence
from torchstream.sliding_window.sliding_window_params import SlidingWindowParams
from torchstream.stream import NotEnoughInputError, Stream


def get_streaming_params(sli_params: SlidingWindowParams):
    # Number of windows that are wasted on the left solely due to padding. "wasted" here means that we recompute
    # these windows on each step despite them being unnecessary, simply because the transform re-pads the input
    # every time. If it is possible to remove padding from the transform and manually pad the streamed input,
    # this waste of compute can be avoided.
    # Note that right padding wastes compute too, but we keep track of that in a different manner.
    n_left_wins_wasted = int(math.ceil(sli_params.left_pad / sli_params.stride_in))
    # For a given output window, the number of other output windows that overlap it. Only >0 when the out stride
    # is smaller than the out kernel size.
    # Note that we need to buffer enough past context in order to have the overlapping windows neccessary in
    # computing a given output. This induces redundant compute that could be avoided if the reduce operation on
    # overlapping windows (e.g. a vector sum) is known. (TODO? implement)
    n_overlapping_out_wins = int(math.ceil(sli_params.kernel_size_out / sli_params.stride_out)) - 1

    n_trimmed_wins = int(math.ceil(sli_params.out_trim / sli_params.stride_out))

    # Number of windows that are needed as extra context on the left
    n_wins_left_context = n_left_wins_wasted + max(n_overlapping_out_wins, n_trimmed_wins)

    # Number of input elements that are needed as extra context on the right
    n_elems_right_context = sli_params.right_pad

    # Bias in computing the effective size of the input sequence
    eff_size_bias = sli_params.kernel_size_in - sli_params.left_pad

    # Slope in computing the effective number of windows
    elem_in_to_win_ratio = sli_params.stride_in

    win_to_elem_out_ratio = sli_params.stride_out

    return n_wins_left_context, n_elems_right_context, eff_size_bias, elem_in_to_win_ratio, win_to_elem_out_ratio


class SlidingWindowStream(Stream):
    def __init__(
        self,
        transform: Callable,
        sliding_window_params: SlidingWindowParams,
        input_spec: SeqSpec,
        output_spec: SeqSpec | None = None,
    ):
        super().__init__(input_spec, output_spec)

        self.transform = transform

        self.params = sliding_window_params
        (
            self.n_wins_left_context,
            self.n_elems_right_context,
            self.eff_size_bias,
            self.elem_in_to_win_ratio,
            self.win_to_elem_out_ratio,
        ) = get_streaming_params(sliding_window_params)
        self.n_wins_to_buffer_left = self.n_wins_left_context

        # Buffer for held back output. This is only returned in the special case where the stream is closed without
        # being to compute any new window, and some previous output has not been returned yet.
        self._prev_trimmed_output = None

    # FIXME: signature
    def _step(self, in_seq: Sequence) -> Union[Tensor, np.ndarray, Tuple[Tensor, np.ndarray]]:
        # Of the windows that will be computed, find the subrange that will effectively be used in the output.
        first_eff_win_idx = self.n_wins_left_context - self.n_wins_to_buffer_left
        last_win_idx = (in_seq.size - self.eff_size_bias + self.n_elems_right_context) // self.elem_in_to_win_ratio
        if in_seq.input_closed:
            last_eff_win_idx = last_win_idx
        else:
            last_eff_win_idx = min(
                # Don't account for windows with right padding, as it will lead to incorrect outputs
                (in_seq.size - self.eff_size_bias) // self.elem_in_to_win_ratio,
                # FIXME! incorrect, too conservative
                last_win_idx - int(math.ceil(self.params.out_trim / self.params.stride_out)),
            )
        eff_num_wins = last_eff_win_idx - first_eff_win_idx + 1

        out_size = last_win_idx * self.params.stride_out + self.params.kernel_size_out - 2 * self.params.out_trim

        out_trim_start = first_eff_win_idx * self.win_to_elem_out_ratio
        if self.n_wins_left_context != self.n_wins_to_buffer_left:
            out_trim_start -= self.params.out_trim
        if in_seq.input_closed:
            out_trim_end = out_size
        else:
            out_trim_end = (last_eff_win_idx + 1) * self.win_to_elem_out_ratio - self.params.out_trim

        if eff_num_wins <= 0 or out_trim_end < out_trim_start:
            if self.input_closed and self._prev_trimmed_output is not None:
                return self._prev_trimmed_output

            # TODO: breakdown current state & display how much more data is needed
            raise NotEnoughInputError(f"Input sequence of size {in_seq.size} is not enough to produce any output.")
        assert out_trim_end > out_trim_start >= 0, "Internal error"

        # Forward the input
        tsfm_out = Sequence.apply(self.transform, in_seq, self.out_spec)
        if tsfm_out.size != out_size:
            raise ValueError(
                f"Sliding window parameters are not matching {self.transform}, got a {tsfm_out.size} sized "
                f"sequence instead of {out_size} for {in_seq.size} sized "
                f"input. Sliding window params: {self.params}"
            )

        # Drop input that won't be necessary in the future
        if in_seq.input_closed:
            in_seq.drop()
        elif eff_num_wins > self.n_wins_to_buffer_left:
            in_seq.drop((eff_num_wins - self.n_wins_to_buffer_left) * self.elem_in_to_win_ratio)
        self.n_wins_to_buffer_left = max(0, self.n_wins_to_buffer_left - eff_num_wins)

        # If we're trimming on the right, save the trim in case the stream closes before we can compute any
        # new sliding window output.
        if out_trim_end and out_trim_end < tsfm_out.size:
            self._prev_trimmed_output = tsfm_out[out_trim_end:]
        else:
            self._prev_trimmed_output = None

        return tsfm_out[out_trim_start:out_trim_end]
