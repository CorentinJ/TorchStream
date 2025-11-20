import itertools
from typing import Callable, Tuple

import numpy as np
import pytest
import torch

from torchstream.sequence.dtype import SeqArrayLike
from torchstream.sequence.sequence import Sequence
from torchstream.sliding_window.nan_trick import get_nan_idx
from torchstream.stream import Stream


@pytest.mark.skip()
@torch.no_grad()
def test_stream_equivalent(
    sync_fn: Callable,
    stream: Stream,
    # TODO: offer comparison to an output array instead, to avoid recomputing for multiple streams
    # TODO: overloads with input sequence size
    in_data: Tuple[SeqArrayLike, ...] | Sequence | None = None,
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
    # Get the input
    if in_data is None:
        in_seq = stream.in_spec.new_randn_sequence(sum(in_step_sizes))
    elif isinstance(in_data, Sequence):
        in_seq = in_data
    else:
        in_seq = stream.in_spec.new_sequence_from_data(*in_data)

    # Get the sync output
    out_ref_seq = in_seq.apply(sync_fn, stream.out_spec)
    if out_ref_seq.size == 0:
        raise ValueError("Input size is too small for the transform to produce any output")

    # FIXME: this is a trivial hack that assumes that the input size is at least the kernel size, ideally we'd only
    # add the kernel size - 1 NaNs to the input.
    in_nan_trick_seq = in_seq.copy()
    in_nan_trick_seq.feed(in_seq)

    step_size_iter = iter(itertools.cycle(in_step_sizes))
    i = 0
    while not stream.is_closed:
        # Read the next input chunk
        step_size = next(step_size_iter)
        in_stream_i = in_seq.read(step_size)

        # Forward through the stream to get an output chunk
        out_seq_stream_i = stream(in_stream_i, is_last_input=not in_seq.size, on_starve="empty")

        # Read the corresponding output chunk from the sync output
        out_sync_i = out_ref_seq.read(out_seq_stream_i.size)

        # Compare the stream to the sync output chunk
        if out_seq_stream_i.size:
            for sync_arr, stream_arr in zip(out_sync_i.data, out_seq_stream_i.data):
                max_error = np.abs(sync_arr - stream_arr).max()
                if max_error > atol or np.isnan(max_error):
                    raise ValueError(
                        f"Error too large on step {i} (got {max_error}, expected <= {atol})\n"
                        f"Sync: {sync_arr}\nStream: {stream_arr}"
                    )

        # Check throughput with the NaN trick
        if throughput_check_max_delay is not None and not stream.is_closed:
            in_nan_trick_seq_i = in_nan_trick_seq.copy()
            in_nan_trick_seq_i[stream.total_in_fed :] = float("nan")
            out_nan_trick_seq_i = in_nan_trick_seq_i.apply(sync_fn, stream.out_spec)
            out_nan_idx = get_nan_idx(out_nan_trick_seq_i)

            # FIXME: handle
            if not len(out_nan_idx):
                raise ValueError("Transform did not output any NaN")
            if out_nan_idx[0] < stream.total_out_produced:
                raise RuntimeError("Internal error: stream has output more than sync")
            if stream.total_out_produced < out_nan_idx[0] - throughput_check_max_delay:
                raise ValueError(
                    f"The stream has output less than what's possible to output based on the NaN trick. "
                    f"Expected {out_nan_idx[0]} outputs total at step {i}, got {stream.total_out_produced} (max delay "
                    f"is {throughput_check_max_delay})"
                )

        i += 1

    if out_ref_seq.size:
        raise ValueError(f"Stream output is too short, {out_ref_seq.size} more outputs were expected")
