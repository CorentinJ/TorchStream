import math
from typing import Callable, Optional, Tuple, Union

import numpy as np
from torch import Tensor

from torchstream.sequence_spec import SeqSpec
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
        # these windows on each step despite them being unnecessary, where proper streaming would not recompute them.
        self._n_left_wins_wasted = int(math.ceil(self.params.left_pad / self.params.stride_in))
        self._n_extra_wins_to_buffer = self._n_left_wins_wasted

    # FIXME: signature
    def _step(self) -> Union[Tensor, np.ndarray, Tuple[Tensor, np.ndarray]]:
        # TODO: multi input support
        in_buff = self._in_buffs[0]

        (left_pad, right_pad), num_wins, expected_out_size = self.params.get_metrics_for_input(in_buff.size)

        # Get the index of the first window that will compute valid output that we'll return
        first_eff_win_idx = self._n_left_wins_wasted - self._n_extra_wins_to_buffer
        out_trim_start = first_eff_win_idx * self.params.stride_out

        # If the input is closed, there is no output trimming that needs to occur on the right side
        if in_buff.input_closed:
            out_trim_end = None
            eff_num_wins = num_wins - first_eff_win_idx
        # Otherwise, we need to trim the output where it would start being incorrect due to the right input padding
        else:
            last_eff_win_idx = int(
                math.ceil(max(0, left_pad + in_buff.size - self.params.kernel_size_in) / self.params.stride_in)
            )
            out_trim_end = last_eff_win_idx * self.params.stride_out
            eff_num_wins = last_eff_win_idx - first_eff_win_idx

        if eff_num_wins <= 0:
            # TODO: breakdown current state & display how much more data is needed
            raise NotEnoughInputError(f"Sequence of size {in_buff.size} is not enough to produce any output.")

        # Forward the input
        tsfm_input = in_buff.peek()
        tsfm_output = self.transform(tsfm_input)
        actual_out_size = self.out_spec.get_seq_size(tsfm_output)
        if actual_out_size != expected_out_size:
            raise ValueError(
                f"Sliding window parameters are not matching {self.transform}, got a {actual_out_size} sized "
                f"sequence instead of {expected_out_size} for {in_buff.size} sized input. Sliding window params: "
                f"{self.params}"
            )

        # Drop input that won't be necessary in the future
        if in_buff.input_closed:
            in_buff._clear_buf()
        elif eff_num_wins > self._n_extra_wins_to_buffer:
            in_buff.drop((eff_num_wins - self._n_extra_wins_to_buffer) * self.params.stride_in)
        self._n_extra_wins_to_buffer = max(0, self._n_extra_wins_to_buffer - eff_num_wins)

        # # We need to discard outputs on the left and right since this is a naive implementation
        # out_buff = StreamBuffer(tsfm_output, dim=-1)
        # out_buff.drop(self._n_left_wins_to_discard * self.params.stride_out)
        # trimmed_output = out_buff.peek((num_wins - self._n_left_wins_to_discard))

        slices = self.out_spec.get_slices(seq_start=out_trim_start, seq_stop=out_trim_end)
        return tsfm_output[slices]
