import itertools
from typing import Callable, Tuple

import numpy as np
import pytest
import torch

from torchstream.sequence.dtype import SeqArrayLike
from torchstream.sequence.sequence import Sequence
from torchstream.sliding_window.nan_trick import get_seq_nan_idx
from torchstream.stream import Stream


@pytest.mark.skip()
@torch.no_grad()
def test_stream_equivalent(
    sync_fn: Callable,
    stream: Stream,
    # TODO: offer comparison to an output array instead, to avoid recomputing for multiple streams
    in_seq: Sequence | SeqArrayLike | None = None,
    in_step_sizes: Tuple[int, ...] = (7, 4, 12, 1, 17, 9),
    atol: float = 1e-5,
    throughput_check_max_delay: int | None = None,
):
    """
    Tests if a stream implementation gives outputs close or equivalent to its synchronous counterpart.

    Both the stream and the sync function must take the same arguments, and return the same number of outputs.
    Outputs must be sequential data of the same shape.
    TODO: better doc

    :param throughput_check_max_delay: TODO: doc
    """
    if in_seq is None:
        in_seq = Sequence.randn(stream.in_spec, seq_size=sum(in_step_sizes))
    elif isinstance(in_seq, Sequence):
        in_seq = in_seq.copy()
    else:
        in_seq = Sequence(stream.in_spec, in_seq, close_input=True)

    # Get the sync output
    out_seq_ref = Sequence.apply(sync_fn, in_seq, stream.out_spec)

    # FIXME: this is a trivial hack that assumes that the input size is at least the kernel size, ideally we'd only
    # add the kernel size - 1 NaNs to the input.
    in_nan_trick_seq = Sequence(in_seq, in_seq, close_input=True)

    step_size_iter = iter(itertools.cycle(in_step_sizes))
    i = 0
    while not stream.output_closed:
        step_size = next(step_size_iter)
        in_stream_i = in_seq.consume(step_size)

        # FIXME: this is a seq
        out_seq_stream_i = stream(in_stream_i, is_last_input=not in_seq.size, on_starve="empty")

        out_sync_i = out_seq_ref.consume(out_seq_stream_i.size)
        total_stream_out = out_seq_ref.n_consumed

        # Ensure the outputs are close
        if out_sync_i.shape != out_seq_stream_i.shape:
            raise ValueError(f"Shape mismatch on step {i} (got {out_seq_stream_i.shape}, expected {out_sync_i.shape})")
        if out_seq_stream_i.size:
            max_error = np.abs(out_sync_i - out_seq_stream_i.data).max()
            if max_error > atol or np.isnan(max_error):
                raise ValueError(
                    f"Error too large on step {i} (got {max_error}, expected <= {atol})\n"
                    f"Sync: {out_sync_i}\nStream: {out_seq_stream_i.data}"
                )

        # Check throughput with the NaN trick
        if throughput_check_max_delay is not None and not stream.output_closed:
            in_nan_trick_seq_i = in_nan_trick_seq.copy()
            in_nan_trick_seq_i[in_seq.n_consumed :] = float("nan")
            out_nan_trick_seq_i = Sequence.apply(sync_fn, in_nan_trick_seq_i, stream.out_spec)
            out_nan_idx = get_seq_nan_idx(out_nan_trick_seq_i)

            # FIXME: handle
            if not len(out_nan_idx):
                raise ValueError("Transform did not output any NaN")
            if out_nan_idx[0] < total_stream_out:
                raise RuntimeError("Internal error: stream has output more than sync")
            if total_stream_out < out_nan_idx[0] - throughput_check_max_delay:
                raise ValueError(
                    f"The stream has output less than what's possible to output based on the NaN trick. "
                    f"Expected {out_nan_idx[0]} outputs total at step {i}, got {total_stream_out} (max delay is "
                    f"{throughput_check_max_delay})"
                )

        i += 1

    if not out_seq_ref.output_closed:
        raise ValueError(f"Stream output is too short, {out_seq_ref.size} more outputs were expected")
