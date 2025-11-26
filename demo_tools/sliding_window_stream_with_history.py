from typing import Callable

from torchstream import SeqSpec, Sequence, SlidingWindowParams, SlidingWindowStream
from torchstream.sliding_window.sliding_window_stream import IncorrectSlidingWindowParametersError
from torchstream.stream import NotEnoughInputError

# TODO: make this a feature of the base class?


class SlidingWindowStreamWithHistory(SlidingWindowStream):
    def __init__(
        self,
        transform: Callable,
        sliding_window_params: SlidingWindowParams,
        input_spec: SeqSpec,
        output_spec: SeqSpec | None = None,
    ):
        super().__init__(transform, sliding_window_params, input_spec, output_spec)

        self.step_history = []

    def _step(self, in_buff: Sequence, is_last_input: bool) -> Sequence:
        step_rec = {}
        step_rec["in_buff_start_pos"] = 0 if not self.step_history else self.step_history[-1]["in_buff_drop_pos"]
        step_rec["in_new_start_pos"] = 0 if not self.step_history else self.step_history[-1]["in_end_pos"]
        step_rec["in_end_pos"] = step_rec["in_buff_start_pos"] + in_buff.size

        # See where the output should be trimmed
        out_size, out_trim_start, out_trim_end = self.get_next_output_slice(in_buff.size, is_last_input)
        step_rec["out_start_pos"] = self.tsfm_out_pos
        step_rec.update(out_size=out_size, out_trim_start=out_trim_start, out_trim_end=out_trim_end)

        if not out_size:
            if is_last_input and self._prev_trimmed_output is not None:
                step_rec["untrimmed_output"] = self._prev_trimmed_output.copy()
                step_rec["in_buff_drop_pos"] = 0 if not self.step_history else self.step_history[-1]["in_buff_drop_pos"]
                self.step_history.append(step_rec)
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
        step_rec["untrimmed_output"] = tsfm_out.copy()

        # Drop input that won't be necessary in the future. We retain only the context size rounded up to the nearest
        # multiple of the input stride.
        wins_to_drop = max(0, (in_buff.size - self.min_buffsize) // self.params.stride_in)
        in_buff.drop(wins_to_drop * self.params.stride_in)
        step_rec["in_buff_drop_pos"] = step_rec["in_buff_start_pos"] + wins_to_drop * self.params.stride_in
        self.step_history.append(step_rec)

        # We've dropped past inputs, so the transform will now produce outputs starting further in the sequence
        self.tsfm_out_pos += wins_to_drop * self.params.stride_out

        # If we're trimming on the right, save the trim in case the stream closes before we can compute any
        # new sliding window output.
        self._prev_trimmed_output = tsfm_out[out_trim_end:] if out_trim_end < out_size else None

        return tsfm_out[out_trim_start:out_trim_end]
