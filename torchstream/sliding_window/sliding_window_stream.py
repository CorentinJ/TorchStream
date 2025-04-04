from typing import Callable, Tuple, Union

import numpy as np
from torch import Tensor

from torchstream.sliding_window.sliding_window_params import SlidingWindowParams
from torchstream.stream import Stream


class SlidingWindowStream(Stream):
    def __init__(
        self,
        transform: Callable,
        sliding_window_params: SlidingWindowParams,
    ):
        # FIXME!
        super().__init__((-1,), (-1,))

        self.transform = transform
        self.params = sliding_window_params

        # self._n_left_wins_to_discard = 0
        # self._input_size_needed = self.params.kernel_size_in - self.params.left_pad
        self._n_eff_wins_out = 0

    def _step(self) -> Union[Tensor, np.ndarray, Tuple[Tensor, np.ndarray]]:
        # TODO: multi input support
        in_buff = self._in_buffs[0]

        # # We need less inputs on the last step, because right padding will then be useful
        # if in_buff.output_closed:
        #     self._input_size_needed -= self.params.right_pad
        # if in_buff.size < self._input_size_needed:
        #     # TODO
        #     raise RuntimeError()

        (left_pad, right_pad), num_wins, expected_out_size = self.params.get_metrics_for_input(in_buff.size)
        if not num_wins:
            raise RuntimeError()

        eff_num_wins = 0
        out_trim_start = None
        out_trim_end = None
        for win_idx in range(num_wins):
            in_sli = slice(
                win_idx * self.params.stride_in, win_idx * self.params.stride_in + self.params.kernel_size_in
            )
            out_sli = slice(
                win_idx * self.params.stride_out, win_idx * self.params.stride_out + self.params.kernel_size_out
            )

            if win_idx >= self._n_eff_wins_out:
                if out_trim_start is None:
                    out_trim_start = out_sli.start

                if not in_buff.input_closed and in_sli.stop >= left_pad + in_buff.size:
                    break

                eff_num_wins += 1
        out_trim_end = out_sli.start
        assert eff_num_wins
        assert out_trim_start < out_trim_end

        # Forward the input
        tsfm_input = in_buff.peek()
        tsfm_output = self.transform(tsfm_input)
        # FIXME: dims
        if tsfm_output.shape[-1] != expected_out_size:
            raise ValueError(
                f"Sliding window parameters are not matching {self.transform}, got a {tsfm_output.shape[-1]} sized "
                f"output instead of {expected_out_size} for {in_buff.size} sized input. Sliding window params: "
                f"{self.params}"
            )

        # # We need to discard outputs on the left and right since this is a naive implementation
        # out_buff = StreamBuffer(tsfm_output, dim=-1)
        # out_buff.drop(self._n_left_wins_to_discard * self.params.stride_out)
        # trimmed_output = out_buff.peek((num_wins - self._n_left_wins_to_discard))

        return tsfm_output[..., out_trim_start:out_trim_end]
