from typing import Callable, Optional, Tuple, Union

import numpy as np
from torch import Tensor

from torchstream.sequence_spec import SeqSpec
from torchstream.sliding_window.sliding_window_params import SlidingWindowParams
from torchstream.stream import NotEnoughInputsError, Stream


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

        # self._n_left_wins_to_discard = 0
        # self._input_size_needed = self.params.kernel_size_in - self.params.left_pad
        self._n_eff_wins_out = 0

    # FIXME: signature
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
        # if not num_wins:
        #     raise NotEnoughInputsError(
        #         f"Need a sequence of size {self.params.get_min_input_size()} to produce any output, "
        #         f"got a sequence of size {in_buff.size}."
        #     )

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

                if not in_buff.input_closed and in_sli.start >= left_pad + in_buff.size:
                    out_trim_end = out_sli.start
                    break

                eff_num_wins += 1

        if not eff_num_wins:
            raise NotEnoughInputsError()

        assert eff_num_wins
        assert out_trim_end is None or out_trim_start < out_trim_end
        self._n_eff_wins_out += eff_num_wins

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

        # # We need to discard outputs on the left and right since this is a naive implementation
        # out_buff = StreamBuffer(tsfm_output, dim=-1)
        # out_buff.drop(self._n_left_wins_to_discard * self.params.stride_out)
        # trimmed_output = out_buff.peek((num_wins - self._n_left_wins_to_discard))

        return tsfm_output[..., out_trim_start:out_trim_end]
