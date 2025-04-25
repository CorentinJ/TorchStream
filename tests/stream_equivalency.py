import itertools
from typing import Callable, Optional, Tuple

import numpy as np
import pytest
import torch

from torchstream.buffers.stream_buffer import StreamBuffer
from torchstream.sequence_spec import Sequence
from torchstream.sliding_window.nan_trick import get_nan_range, set_nan_range
from torchstream.stream import NotEnoughInputError, Stream


@pytest.mark.skip("Not a test")
@torch.no_grad()
def test_stream_equivalent(
    sync_fn: Callable,
    stream: Stream,
    input_provider: Optional[Callable[[int], Sequence]] = None,
    in_step_sizes: Tuple[int, ...] = (7, 4, 12, 1, 17),
    in_seq_size: int = 50,
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
    input_provider = input_provider or stream.in_spec.randn
    sync_input = input_provider(in_seq_size)

    ## Get the sync output
    out_sync = sync_fn(sync_input)
    out_sync_buff = StreamBuffer(stream.out_spec)
    out_sync_buff.feed(out_sync, close_input=True)
    out_size = out_sync_buff.size

    ## Get the stream output & compare it
    in_buff = StreamBuffer(stream.in_spec)
    in_buff.feed(sync_input, close_input=True)
    in_size = in_buff.size

    # TODO: cleaner implementation of the throughput trick
    stream_fed_seq_size = 0
    nan_trick_in_buff = StreamBuffer(stream.in_spec)
    nan_trick_in_buff.feed(sync_input)
    # FIXME: this is a trivial hack that assumes that the input size at least the kernel size, ideally we'd only
    # add the kernel size - 1 NaNs to the input.
    nan_trick_in_buff.feed(sync_input, close_input=True)
    nan_trick_in = nan_trick_in_buff.read()
    set_nan_range(nan_trick_in, (in_size, None), dim=stream.in_spec.seq_dim)

    step_size_iter = iter(itertools.cycle(in_step_sizes))
    i = 0
    while not stream.output_closed:
        step_size = next(step_size_iter)
        stream_input_i = in_buff.read(step_size)

        try:
            out_stream_i = stream(stream_input_i, is_last_input=in_buff.output_closed)
        except NotEnoughInputError:
            out_stream_i = stream.out_spec.empty()

        stream_fed_seq_size += in_buff.spec.get_seq_size(stream_input_i)

        stream_out_size = out_sync_buff.spec.get_seq_size(out_stream_i)
        out_sync_i = out_sync_buff.read(stream_out_size)

        # Ensure the outputs are close
        assert out_sync_i.shape == out_stream_i.shape, (
            f"Shape mismatch on step {i} (got {out_stream_i.shape}, expected {out_sync_i.shape})"
        )
        if stream_out_size:
            max_error = np.abs(out_sync_i - out_stream_i).max()
            assert max_error <= atol, (
                f"Error too large on step {i} (got {max_error}, expected <= {atol})\n"
                f"Sync: {out_sync_i}\nStream: {out_stream_i}"
            )

        # Check throughput with the NaN trick
        if check_throughput_with_nan_trick and not stream.output_closed:
            # TODO: this api is terrible, let's improve it
            copy_fn = np.copy if stream.in_spec.is_numpy else torch.clone
            nan_trick_in_i = copy_fn(nan_trick_in)
            set_nan_range(nan_trick_in_i, (stream_fed_seq_size, in_size), dim=stream.in_spec.seq_dim)
            out_nan_trick_i = sync_fn(nan_trick_in_i)
            out_nan_size = out_sync_buff.spec.get_seq_size(out_nan_trick_i)
            nan_range = get_nan_range(out_nan_trick_i, dim=stream.out_spec.seq_dim)
            assert nan_range, "Internal error: kernel size must be greater than the input sequence size"

            assert nan_range[1] == out_nan_size, (
                f"Transform is not suitable for NaN trick, NaNs set at {(stream_fed_seq_size, 2 * in_size)} in the "
                f"input propagated up to {nan_range}, when the output is {out_nan_size} long. NaNs should have "
                f"propagated to the end."
            )

            current_output_size = out_size - out_sync_buff.size
            assert nan_range[0] >= current_output_size, "Internal error"
            assert nan_range[0] == current_output_size, (
                f"The stream has output less than what's possible to output based on the NaN trick's output. "
                f"Expected {nan_range[0]} outputs total at step {i}, got {current_output_size}."
            )

        i += 1

    assert out_sync_buff.output_closed, f"Stream output is too short, {out_sync_buff.size} more outputs expected"
