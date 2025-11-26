from typing import Callable, Tuple

from torchstream.sequence.sequence import SeqSpec, Sequence
from torchstream.sliding_window.sliding_window_params import SlidingWindowParams, get_output_delay
from torchstream.stream import NotEnoughInputError, Stream


class IncorrectSlidingWindowParametersError(Exception):
    """
    TODO: doc
    """

    pass


class SlidingWindowStream(Stream):
    """
    TODO!: doc!

    TODO: create_from_kernel vs create from sliding window transform with known parameters
    """

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

        # Buffer for held back output. This is only returned in the special case where the stream is closed without
        # being requested to compute any new window, and some previous output has not been returned yet.
        self._prev_trimmed_output = None

    def get_next_output_slice(self, in_buff_size: int, is_last_input: bool) -> Tuple[int | None, int, int]:
        """
        Given an input buffer size and whether this is the last input, returns the output size and the start and end
        indices of the next output to compute.
        The function will return None in place of the output size if no new output can be computed from the given
        input buffer, implying that the transform should not be run at this time.
        """
        out_size = self.params.get_out_size_for_in_size(in_buff_size)
        if is_last_input:
            out_trim_end = out_size
        else:
            out_delay = get_output_delay(self.params, in_buff_size)
            out_trim_end = max(out_size - out_delay, 0)

        if self.tsfm_out_pos + out_trim_end <= self.total_out_produced:
            out_size = None

        out_trim_start = self.total_out_produced - self.tsfm_out_pos
        assert (not out_size) or out_trim_end > out_trim_start >= 0, "Internal error"

        return out_size, out_trim_start, out_trim_end

    def _step(self, in_buff: Sequence, is_last_input: bool) -> Sequence:
        # See where the output should be trimmed
        out_size, out_trim_start, out_trim_end = self.get_next_output_slice(in_buff.size, is_last_input)

        if not out_size:
            if is_last_input and self._prev_trimmed_output is not None:
                return self._prev_trimmed_output

            # TODO: breakdown current state & display how much more data is needed
            raise NotEnoughInputError(f"Input sequence of size {in_buff.size} is not enough to produce any output.")

        # Forward the input
        tsfm_out = in_buff.apply(self.transform, self.out_spec)
        if tsfm_out.size != out_size:
            raise IncorrectSlidingWindowParametersError(
                f"Sliding window parameters are not matching {self.transform}, got a {tsfm_out.size} sized "
                f"sequence instead of {out_size} for {in_buff.size} sized input."
            )

        # Drop input that won't be necessary in the future. We retain only the context size rounded up to the nearest
        # multiple of the input stride.
        wins_to_drop = max(0, (in_buff.size - self.min_buffsize) // self.params.stride_in)
        in_buff.drop(wins_to_drop * self.params.stride_in)

        # We've dropped past inputs, so the transform will now produce outputs starting further in the sequence
        self.tsfm_out_pos += wins_to_drop * self.params.stride_out

        # If we're trimming on the right, save the trim in case the stream closes before we can compute any
        # new sliding window output.
        self._prev_trimmed_output = tsfm_out[out_trim_end:] if out_trim_end < out_size else None

        return tsfm_out[out_trim_start:out_trim_end]
