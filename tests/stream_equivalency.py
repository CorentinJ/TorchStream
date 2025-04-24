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
    nan_trick_max_in_kernel_gap: int = 0,
):
    """
    Tests if a stream implementation gives outputs close or equivalent to its synchronous counterpart.

    Both the stream and the sync function must take the same arguments, and return the same number of outputs.
    Outputs must be sequential data of the same shape.
    TODO: better doc

    :param check_throughput_with_nan_trick: TODO
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

    step_size_iter = iter(itertools.cycle(in_step_sizes))
    i = 0
    # TODO: cleaner implementation of the throughput trick
    stream_fed_seq_size = 0
    while not stream.output_closed:
        step_size = next(step_size_iter)
        stream_input_i = in_buff.read(step_size)

        try:
            out_stream_i = stream(stream_input_i, is_last_input=in_buff.output_closed)
        except NotEnoughInputError:
            out_stream_i = stream.out_spec.empty()

        stream_fed_seq_size += step_size

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
        # Note that the NaN trick is not guaranteed to work if the kernel has gaps larger than the number of
        # consecutive NaNs; in that case we skip the check.
        if check_throughput_with_nan_trick and in_size - stream_fed_seq_size > nan_trick_max_in_kernel_gap:
            copy_fn = np.copy if stream.in_spec.is_numpy else torch.clone
            sync_input_i = copy_fn(sync_input)
            set_nan_range(sync_input_i, (stream_fed_seq_size, in_size), dim=stream.in_spec.seq_dim)
            out_sync_i = sync_fn(sync_input_i)
            nan_range = get_nan_range(out_sync_i, dim=stream.out_spec.seq_dim)

            # FIXME
            nan_range = nan_range or (out_size, out_size)

            assert nan_range[1] == out_size, (
                f"Transform is not suitable for NaN trick, NaNs set at {(stream_fed_seq_size, in_size)} in the input "
                f"propagated up to {nan_range}, when the output is {out_size} long. NaNs should have propagated to "
                f"the end."
            )

            current_output_size = out_size - out_sync_buff.size
            assert nan_range[0] >= current_output_size, "Internal error"
            assert nan_range[0] == current_output_size, (
                f"The stream has output less than what's possible to output based on the NaN trick's output. "
                f"Expected {nan_range[0]} outputs total at step {i}, got {current_output_size}."
            )

        i += 1

    assert out_sync_buff.output_closed, f"Stream output is too short, {out_sync_buff.size} outputs remain"
