from typing import Callable, Tuple, Union

import numpy as np
from torch import Tensor

from torchstream.sequence.seq_spec import SeqSpec
from torchstream.sequence.sequence import Sequence
from torchstream.sliding_window.sliding_window_params import SlidingWindowParams
from torchstream.stream import NotEnoughInputError, Stream


class IncorrectSlidingWindowParametersError(Exception):
    """
    TODO: doc
    """

    pass


class SlidingWindowStream(Stream):
    def __init__(
        self,
        transform: Callable,
        # TODO: class for sliding window stream params
        sliding_window_params: SlidingWindowParams | Tuple,
        input_spec: SeqSpec,
        output_spec: SeqSpec | None = None,
    ):
        super().__init__(input_spec, output_spec)

        self.transform = transform

        # FIXME!!
        self.s = sliding_window_params

        (
            self.stride_in,
            self.stride_out,
            self.in_size_bias,
            self.out_size_bias,
            self.in_delay,
            self.out_delay,
            self.in_context_size,
        ) = (
            get_canonicalized_in_out_size_biases(sliding_window_params)
            if isinstance(sliding_window_params, SlidingWindowParams)
            else sliding_window_params
        )

        self.tsfm_out_pos = 0
        self.stream_out_pos = 0

        # Buffer for held back output. This is only returned in the special case where the stream is closed without
        # being requested to compute any new window, and some previous output has not been returned yet.
        self._prev_trimmed_output = None

    # FIXME: signature
    def _step(self, in_seq: Sequence) -> Union[Tensor, np.ndarray, Tuple[Tensor, np.ndarray]]:
        # Compute the actual output size we'll get from the transform
        out_size_t1 = (in_seq.size + self.in_size_bias) // self.stride_in
        out_size = max(0, out_size_t1 * self.stride_out + self.out_size_bias)
        sufficient_input = in_seq.size and out_size

        # See where the output should be trimmed
        if in_seq.input_closed:
            out_trim_end = out_size
        elif in_seq.size + self.s.left_pad >= self.s.kernel_size_in:
            t2 = (
                (self.s.left_pad + in_seq.size - self.s.kernel_size_in) % self.s.stride_in + self.s.right_pad
            ) // self.s.stride_in
            tel = self.s.kernel_size_out + (t2 - 1) * self.s.stride_out
            offset2 = max(0, tel - self.s.out_trim)
            out_trim_end = max(out_size - offset2, 0)
        else:
            out_trim_end = 0

        if not sufficient_input or self.tsfm_out_pos + out_trim_end <= self.stream_out_pos:
            if self.input_closed and self._prev_trimmed_output is not None:
                return self._prev_trimmed_output

            # TODO: breakdown current state & display how much more data is needed
            raise NotEnoughInputError(f"Input sequence of size {in_seq.size} is not enough to produce any output.")

        # Forward the input
        tsfm_out = Sequence.apply(self.transform, in_seq, self.out_spec)
        if tsfm_out.size != out_size:
            raise IncorrectSlidingWindowParametersError(
                f"Sliding window parameters are not matching {self.transform}, got a {tsfm_out.size} sized "
                f"sequence instead of {out_size} for {in_seq.size} sized input."
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
