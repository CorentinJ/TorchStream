from typing import Callable, Tuple, Union

import numpy as np
from torch import Tensor

from torchstream.sequence.seq_spec import SeqSpec
from torchstream.sequence.sequence import Sequence
from torchstream.sliding_window.sliding_window_params import SlidingWindowParams
from torchstream.sliding_window.sliding_window_stream_params import get_streaming_params
from torchstream.stream import NotEnoughInputError, Stream


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
            self.stride_in,
            self.stride_out,
            self.in_delay,
            self.out_delay,
            self.in_context_size,
        ) = get_streaming_params(sliding_window_params)

        self.tsfm_out_pos = 0
        self.stream_out_pos = 0

        # Buffer for held back output. This is only returned in the special case where the stream is closed without
        # being to compute any new window, and some previous output has not been returned yet.
        self._prev_trimmed_output = None

    # FIXME: signature
    def _step(self, in_seq: Sequence) -> Union[Tensor, np.ndarray, Tuple[Tensor, np.ndarray]]:
        # FIXME: signature
        out_size = self.params.get_metrics_for_input(in_seq.size)[2]

        if in_seq.input_closed:
            out_trim_end = out_size
        else:
            last_eff_win_idx = (in_seq.size - self.in_delay) // self.stride_in
            out_trim_end = min((last_eff_win_idx + 1) * self.stride_out - self.out_delay, out_size)

        if in_seq.size < self.params.get_min_input_size() or self.tsfm_out_pos + out_trim_end <= self.stream_out_pos:
            if self.input_closed and self._prev_trimmed_output is not None:
                return self._prev_trimmed_output

            # TODO: breakdown current state & display how much more data is needed
            raise NotEnoughInputError(f"Input sequence of size {in_seq.size} is not enough to produce any output.")

        # Forward the input
        tsfm_out = Sequence.apply(self.transform, in_seq, self.out_spec)
        if tsfm_out.size != out_size:
            raise ValueError(
                f"Sliding window parameters are not matching {self.transform}, got a {tsfm_out.size} sized "
                f"sequence instead of {out_size} for {in_seq.size} sized "
                f"input. Sliding window params: {self.params}"
            )

        # Compute the slice of the output that we'll return and update the stream position
        out_trim_start = self.stream_out_pos - self.tsfm_out_pos
        assert out_trim_end > out_trim_start >= 0, "Internal error"
        self.stream_out_pos = self.tsfm_out_pos + out_trim_end

        # Drop input that won't be necessary in the future
        wins_to_drop = max(0, (in_seq.size - self.in_context_size) // self.stride_in)
        in_seq.drop(wins_to_drop * self.stride_in)
        self.tsfm_out_pos += wins_to_drop * self.stride_out

        # If we're trimming on the right, save the trim in case the stream closes before we can compute any
        # new sliding window output.
        self._prev_trimmed_output = tsfm_out[out_trim_end:] if out_trim_end < out_size else None

        return tsfm_out[out_trim_start:out_trim_end]
