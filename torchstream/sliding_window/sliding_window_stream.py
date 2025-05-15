import math
from typing import Callable, Optional, Tuple, Union

import numpy as np
from torch import Tensor

from torchstream.sequence.seq_spec import SeqSpec
from torchstream.sequence.sequence import Sequence
from torchstream.sliding_window.sliding_window_params import SlidingWindowParams
from torchstream.stream import NotEnoughInputError, Stream


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

        self._out_idx = 0

    # FIXME: signature
    def _step(self, in_seq: Sequence) -> Union[Tensor, np.ndarray, Tuple[Tensor, np.ndarray]]:
        if in_seq.input_closed:
            inv_map = self.params.get_inverse_map(in_seq.size)
            eff_stop = None
        else:
            # NOTE: the kernel size here is the kernel span
            inv_map = self.params.get_inverse_map(
                input_size=in_seq.size + self.params.kernel_size_in - 1,
                limit_to_input_bounds=False,
            )
            eff_stop = np.searchsorted(inv_map[:, 1], in_seq.size, side="right")

        out_trim_start, out_trim_end, _ = slice.indices(slice(self._out_idx, eff_stop), len(inv_map))
        self._out_idx = out_trim_end

        # if eff_num_wins <= 0:
        # if self.input_closed and self._prev_trimmed_output is not None:
        # return self._prev_trimmed_output
        if out_trim_start == out_trim_end:
            # TODO: breakdown current state & display how much more data is needed
            raise NotEnoughInputError(f"Input sequence of size {in_seq.size} is not enough to produce any output.")

        # Forward the input
        tsfm_out = Sequence.apply(self.transform, in_seq, self.out_spec)

        # # Drop input that won't be necessary in the future
        # if in_seq.input_closed:
        #     in_seq.drop()
        # elif eff_num_wins > self._n_wins_to_buffer_left:
        #     in_seq.drop((eff_num_wins - self._n_wins_to_buffer_left) * self.params.stride_in)
        # self._n_wins_to_buffer_left = max(0, self._n_wins_to_buffer_left - eff_num_wins)

        # # If we're trimming on the right, save the trim in case the stream closes before we can compute any
        # # new sliding window output.
        # if out_trim_end and out_trim_end < tsfm_out.size:
        #     self._prev_trimmed_output = tsfm_out[out_trim_end:]
        # else:
        #     self._prev_trimmed_output = None

        return tsfm_out[out_trim_start:out_trim_end]

    # FIXME: signature
    def _step2(self, in_seq: Sequence) -> Union[Tensor, np.ndarray, Tuple[Tensor, np.ndarray]]:
        (left_pad, right_pad), num_wins, expected_out_size = self.params.get_metrics_for_input(in_seq.size)

        # Get the index of the first window that will compute valid output that we'll return
        first_eff_win_idx = self._n_wins_left_context - self._n_wins_to_buffer_left
        out_trim_start = first_eff_win_idx * self.params.stride_out

        # Likewise, get the index of the last window with valid output
        # If the input is closed, there is no output trimming that needs to occur on the right side
        if in_seq.input_closed:
            out_trim_end = None
            eff_num_wins = num_wins - first_eff_win_idx
        # Otherwise, we may need to trim output on the right due to erroneous inputs incurred by right padding
        else:
            num_wins_before_right_trim = num_wins - max(0, int(math.ceil((right_pad / self.params.stride_in))))
            out_trim_end = num_wins_before_right_trim * self.params.stride_out
            eff_num_wins = num_wins_before_right_trim - first_eff_win_idx

        if eff_num_wins <= 0:
            if self.input_closed and self._prev_trimmed_output is not None:
                return self._prev_trimmed_output

            # TODO: breakdown current state & display how much more data is needed
            raise NotEnoughInputError(f"Input sequence of size {in_seq.size} is not enough to produce any output.")

        # Forward the input
        tsfm_out = Sequence.apply(self.transform, in_seq, self.out_spec)
        if tsfm_out.size != expected_out_size:
            raise ValueError(
                f"Sliding window parameters are not matching {self.transform}, got a {tsfm_out.size} sized "
                f"sequence instead of {expected_out_size} for {in_seq.size} sized input. Sliding window params: "
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
