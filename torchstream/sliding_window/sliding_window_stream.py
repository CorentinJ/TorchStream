import math
from typing import Callable, Optional, Tuple, Union

import numpy as np
from torch import Tensor

from torchstream.sequence.seq_spec import SeqSpec
from torchstream.sequence.sequence import Sequence
from torchstream.sliding_window.sliding_window_params import SlidingWindowParams
from torchstream.stream import NotEnoughInputError, Stream


# FIXME!
def requires_same_context(sli1: SlidingWindowParams, sli2: SlidingWindowParams) -> bool:
    return (
        sli1.kernel_size_in == sli2.kernel_size_in
        and sli1.kernel_size_out == sli2.kernel_size_out
        and sli1.stride_in == sli2.stride_in
        and sli1.left_pad == sli2.left_pad
        and sli1.right_pad == sli2.right_pad
        and sli1.stride_out == sli2.stride_out
        and sli1.out_trim == sli2.out_trim
    )


class SlidingWindowStream(Stream):
    def __init__(
        self,
        transform: Callable,
        sliding_window_params: SlidingWindowParams,
        input_spec: SeqSpec,
        output_spec: Optional[SeqSpec] = None,
    ):
        super().__init__(input_spec, output_spec)

        self.transform = transform
        self.params = sliding_window_params
        assert not self.params.out_trim, "Not implemented"

        # Number of windows that are wasted on the left solely due to padding. "wasted" here means that we recompute
        # these windows on each step despite them being unnecessary, simply because the transform re-pads the input
        # every time. If it is possible to remove padding from the transform and manually pad the streamed input,
        # this waste of compute can be avoided.
        # Note that right padding wastes compute too, but we keep track of that in a different manner.
        self._n_left_wins_wasted = int(math.ceil(self.params.left_pad / self.params.stride_in))
        # For a given output window, the number of other output windows that overlap it. Only >0 when the out stride
        # is smaller than the out kernel size.
        # Note that we need to buffer enough past context in order to have the overlapping windows neccessary in
        # computing a given output. This induces redundant compute that could be avoided if the reduce operation on
        # overlapping windows (e.g. a vector sum) is known. (TODO? implement)
        self._n_overlapping_out_wins = int(math.ceil(self.params.kernel_size_out / self.params.stride_out)) - 1
        self._n_wins_left_context = self._n_left_wins_wasted + self._n_overlapping_out_wins
        self._n_wins_to_buffer_left = self._n_wins_left_context

        # Buffer for held back output. This is only returned in the special case where the stream is closed without
        # being to compute any new window, and some previous output has not been returned yet.
        self._prev_trimmed_output = None

    # FIXME: signature
    def _step(self, in_seq: Sequence) -> Union[Tensor, np.ndarray, Tuple[Tensor, np.ndarray]]:
        # Get the index of the first window that will compute valid output that we'll return
        first_eff_win_idx = self._n_wins_left_context - self._n_wins_to_buffer_left
        right_context_size = self.params.right_pad if in_seq.input_closed else 0
        last_eff_win_idx = (
            self.params.left_pad + in_seq.size + right_context_size - self.params.kernel_size_in
        ) // self.params.stride_in
        eff_num_wins = last_eff_win_idx - first_eff_win_idx + 1

        if eff_num_wins <= 0:
            if self.input_closed and self._prev_trimmed_output is not None:
                return self._prev_trimmed_output

            # TODO: breakdown current state & display how much more data is needed
            raise NotEnoughInputError(f"Input sequence of size {in_seq.size} is not enough to produce any output.")

        out_trim_start = first_eff_win_idx * self.params.stride_out
        out_trim_end = None if in_seq.input_closed else (last_eff_win_idx + 1) * self.params.stride_out
        assert out_trim_start >= 0 and (out_trim_end is None or out_trim_end > out_trim_start), "Internal error"

        # Forward the input
        tsfm_out = Sequence.apply(self.transform, in_seq, self.out_spec)
        # FIXME!
        if tsfm_out.size != self.params.get_metrics_for_input(in_seq.size)[2]:
            raise ValueError(
                f"Sliding window parameters are not matching {self.transform}, got a {tsfm_out.size} sized "
                f"sequence instead of {self.params.get_metrics_for_input(in_seq.size)[2]} for {in_seq.size} sized input. Sliding window params: "
                f"{self.params}"
            )

        # Drop input that won't be necessary in the future
        if in_seq.input_closed:
            in_seq.drop()
        elif eff_num_wins > self._n_wins_to_buffer_left:
            in_seq.drop((eff_num_wins - self._n_wins_to_buffer_left) * self.params.stride_in)
        self._n_wins_to_buffer_left = max(0, self._n_wins_to_buffer_left - eff_num_wins)

        # If we're trimming on the right, save the trim in case the stream closes before we can compute any
        # new sliding window output.
        if out_trim_end and out_trim_end < tsfm_out.size:
            self._prev_trimmed_output = tsfm_out[out_trim_end:]
        else:
            self._prev_trimmed_output = None

        return tsfm_out[out_trim_start:out_trim_end]
