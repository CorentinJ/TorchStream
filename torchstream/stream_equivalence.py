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
    in_arrs: Tuple[Sequence | SeqArrayLike, ...] | None = None,
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
    if in_arrs is None:
        in_buffs = stream.in_spec.new_randn_buffers(sum(in_step_sizes))
    else:
        in_buffs = stream.in_spec.new_buffers_from_data(*in_arrs)

    # Get the sync output
    out_ref_buffs = stream.in_spec.apply(sync_fn, *in_buffs, out_spec=stream.out_spec, to_buffers=True)
    if any(size == 0 for size in out_ref_buffs.size):
        raise ValueError("Input size is too small for the transform to produce any output")

    # FIXME: this is a trivial hack that assumes that the input size is at least the kernel size, ideally we'd only
    # add the kernel size - 1 NaNs to the input.
    in_nan_trick_seq = Sequence(in_seq, in_seq)

    step_size_iter = iter(itertools.cycle(in_step_sizes))
    i = 0
    while not stream.output_closed:
        step_size = next(step_size_iter)
        in_stream_i = in_seq.read(step_size)

        # FIXME: this is a seq
        out_seq_stream_i = stream(in_stream_i, is_last_input=not in_seq.size, on_starve="empty")

        out_sync_i = out_ref_buffs.read(out_seq_stream_i.size)
        total_stream_out = out_ref_buffs.n_consumed

        # Ensure the outputs are close
        if out_sync_i.shape != out_seq_stream_i.shapes:
            raise ValueError(f"Shape mismatch on step {i} (got {out_seq_stream_i.shapes}, expected {out_sync_i.shape})")
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
            out_nan_idx = get_nan_idx(out_nan_trick_seq_i)

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

    if not out_ref_buffs.output_closed:
        raise ValueError(f"Stream output is too short, {out_ref_buffs.size} more outputs were expected")
