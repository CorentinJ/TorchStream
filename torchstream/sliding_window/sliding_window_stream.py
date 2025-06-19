import math
from typing import Callable, Tuple, Union

import numpy as np
from torch import Tensor

from torchstream.sequence.seq_spec import SeqSpec
from torchstream.sequence.sequence import Sequence
from torchstream.sliding_window.sliding_window_params import SlidingWindowParams
from torchstream.stream import NotEnoughInputError, Stream


def get_streaming_params(sli_params: SlidingWindowParams):
    """
    Derives parameters necessary for streaming from the sliding window parameters. Multiple sliding window parameters
    can give rise to the same streaming parameters. Also, incorrect sliding window parameters can give rise to correct
    but suboptimal streaming parameters that use too much context.

    This function returns 5 parameters:
    - stride_in: the stride (reduction factor) for the input sequence
    - stride_out: the stride (multiplication factor) for the output sequence
    - in_offset: offset for the input sequence
    - out_offset: offset for the output sequence
    - in_context_size: number of input elements to be buffered as context
    """
    # These parameters offset the effective size of the input sequence
    in_offset = sli_params.kernel_size_in - sli_params.left_pad

    # Out trimming also offsets the output sequence
    out_offset = sli_params.out_trim

    # Number of windows that are wasted on the left solely due to padding. "wasted" here means that we recompute
    # these windows on each step despite them being unnecessary, simply because the transform re-pads the input
    # every time. If it is possible to remove padding from the transform and manually pad the streamed input,
    # this waste of compute can be avoided.
    # Note that right padding wastes compute just as much, however it does not require any context to be stored.
    n_left_wins_wasted = int(math.ceil(sli_params.left_pad / sli_params.stride_in))

    # For a given output window, the number of other output windows that overlap it. Only >0 when the out stride
    # is smaller than the out kernel size.
    # Note that we need to buffer enough past context in order to have the overlapping windows neccessary in
    # computing a given output. This induces redundant compute that could be avoided if the reduce operation on
    # overlapping windows (e.g. a vector sum) is known. (TODO? implement)
    n_overlapping_out_wins = int(math.ceil(sli_params.kernel_size_out / sli_params.stride_out)) - 1

    # Extra windows necessary to make up for windows lost on the left due to output trimming
    n_trimmed_wins = int(math.ceil(sli_params.out_trim / sli_params.stride_out))

    # Number of windows that are needed as context
    windows_context_size = n_left_wins_wasted + max(n_overlapping_out_wins, n_trimmed_wins)

    # Extra input context necessary to make up for windows lost on the right due to output trimming
    extra_right_context = max(
        0,
        # TODO! verify this is correct with both ki>1 & to>1
        int(math.ceil(sli_params.out_trim / sli_params.stride_out)) * sli_params.stride_in - sli_params.right_pad,
    )

    # Number of input elements that are needed as context
    in_context_size = max(0, (windows_context_size - 1) * sli_params.stride_in + in_offset + extra_right_context)

    return sli_params.stride_in, sli_params.stride_out, in_offset, out_offset, in_context_size


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
            self.in_offset,
            self.out_offset,
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
            last_eff_win_idx = (in_seq.size - self.in_offset) // self.stride_in
            out_trim_end = min((last_eff_win_idx + 1) * self.stride_out - self.out_offset, out_size)

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
