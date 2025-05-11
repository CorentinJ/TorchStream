import itertools
from typing import Callable, Tuple

import numpy as np
import pytest
import torch

from torchstream.sequence.sequence import Sequence
from torchstream.sliding_window.nan_trick import get_nan_range
from torchstream.stream import Stream


@pytest.mark.skip("Not a test")
@torch.no_grad()
def test_stream_equivalent(
    sync_fn: Callable,
    stream: Stream,
    in_seq: Sequence | None = None,
    in_step_sizes: Tuple[int, ...] = (7, 4, 12, 1, 17),
    atol: float = 1e-5,
    check_throughput_with_nan_trick: bool = False,
):
    """
    Tests if a stream implementation gives outputs close or equivalent to its synchronous counterpart.

    Both the stream and the sync function must take the same arguments, and return the same number of outputs.
    Outputs must be sequential data of the same shape.
    TODO: better doc

    :param check_throughput_with_nan_trick: TODO: doc
    """
    if not in_seq:
        in_seq = Sequence.randn(stream.in_spec, seq_size=50)

    # Get the sync output
    out_seq_ref = Sequence.apply(sync_fn, in_seq, stream.out_spec)

    # FIXME: this is a trivial hack that assumes that the input size at least the kernel size, ideally we'd only
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

        # Ensure the outputs are close
        assert out_sync_i.shape == out_seq_stream_i.shape, (
            f"Shape mismatch on step {i} (got {out_seq_stream_i.shape}, expected {out_sync_i.shape})"
        )
        if out_seq_stream_i.size:
            max_error = np.abs(out_sync_i - out_seq_stream_i.data).max()
            assert max_error <= atol, (
                f"Error too large on step {i} (got {max_error}, expected <= {atol})\n"
                f"Sync: {out_sync_i}\nStream: {out_seq_stream_i.data}"
            )

        # Check throughput with the NaN trick
        if check_throughput_with_nan_trick and not stream.output_closed:
            in_nan_trick_seq_i = in_nan_trick_seq.copy()
            in_nan_trick_seq_i[in_seq.consumed :] = float("nan")
            out_nan_trick_seq_i = Sequence.apply(sync_fn, in_nan_trick_seq_i, stream.out_spec)
            nan_range = get_nan_range(out_nan_trick_seq_i)
            assert nan_range, "Internal error: kernel size must be greater than the input sequence size"

            assert nan_range[1] == out_nan_trick_seq_i.size, (
                # FIXME: can also happen if kernel is greater than the input size
                f"Transform is not suitable for NaN trick, NaNs set at {(in_seq.consumed, in_nan_trick_seq.size)} in "
                f"the input propagated up to {nan_range}, when the output is {out_nan_trick_seq_i.size} long. NaNs "
                f"should have propagated to the end."
            )

            assert nan_range[0] >= out_seq_ref.consumed, "Internal error"
            assert nan_range[0] == out_seq_ref.consumed, (
                f"The stream has output less than what's possible to output based on the NaN trick's output. "
                f"Expected {nan_range[0]} outputs total at step {i}, got {out_seq_ref.consumed}."
            )

        i += 1

    assert out_seq_ref.output_closed, f"Stream output is too short, {out_seq_ref.size} more outputs were expected"
