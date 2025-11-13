from typing import Callable

from torchstream.sequence.dtype import SeqArrayLike
from torchstream.sequence.sequence import SeqSpec
from torchstream.sequence.sequence import Sequence
from torchstream.sliding_window.sliding_window_params import SlidingWindowParams, get_output_delay
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
        sliding_window_params: SlidingWindowParams,
        input_spec: SeqSpec,
        output_spec: SeqSpec | None = None,
    ):
        super().__init__(input_spec, output_spec)

        self.transform = transform

        self.params = sliding_window_params
        self.min_buffsize = max(sliding_window_params.streaming_context_size, sliding_window_params.min_input_size - 1)

        self.tsfm_out_pos = 0
        self.stream_out_pos = 0

        # Buffer for held back output. This is only returned in the special case where the stream is closed without
        # being requested to compute any new window, and some previous output has not been returned yet.
        self._prev_trimmed_output = None

    # FIXME: signature
    def _step(self, in_seq: Sequence) -> SeqArrayLike:
        # Compute the actual output size we'll get from the transform
        (_, right_pad), num_wins, out_size = self.params.get_metrics_for_input(in_seq.size)
        sufficient_input = in_seq.size and out_size

        # See where the output should be trimmed
        if self.input_closed:
            out_trim_end = out_size
        else:
            out_delay = get_output_delay(self.params, in_seq.size)
            out_trim_end = max(out_size - out_delay, 0)

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

        # Drop input that won't be necessary in the future. We retain only the context size rounded up to the nearest
        # multiple of the input stride.
        wins_to_drop = max(0, (in_seq.size - self.min_buffsize) // self.params.stride_in)
        in_seq.drop(wins_to_drop * self.params.stride_in)
        self.tsfm_out_pos += wins_to_drop * self.params.stride_out

        # If we're trimming on the right, save the trim in case the stream closes before we can compute any
        # new sliding window output.
        self._prev_trimmed_output = tsfm_out[out_trim_end:] if out_trim_end < out_size else None

        return tsfm_out[out_trim_start:out_trim_end]
