import itertools
from typing import Callable, Optional, Tuple

import numpy as np
import pytest
import torch

from tests.rng import set_seed
from torchstream.buffers.stream_buffer import StreamBuffer
from torchstream.sequence_spec import Sequence
from torchstream.stream import NotEnoughInputError, Stream


@pytest.mark.skip("Not a test")
@torch.no_grad()
def test_stream_equivalent(
    stream: Stream,
    sync_fn: Callable,
    input_provider: Optional[Callable[[int], Sequence]] = None,
    in_step_sizes: Tuple[int, ...] = (7, 4, 12, 1, 17),
    in_seq_size: int = 50,
    atol: float = 1e-5,
    seed=None,
):
    """
    Tests if a stream implementation gives outputs close or equivalent to its synchronous counterpart.

    Both the stream and the sync function must take the same arguments, and return the same number of outputs.
    Outputs must be sequential data of the same shape.
    TODO: better doc

    """
    if seed:
        set_seed(seed)

    input_provider = input_provider or stream.in_spec.randn
    inputs = input_provider(in_seq_size)
    # FIXME
    inputs = (inputs,)

    ## Get the sync output
    out_sync = sync_fn(*inputs)
    # FIXME
    out_sync = (out_sync,)
    assert len(out_sync) == 1, f"The sync function returned {len(out_sync)} outputs, expected {1} outputs"
    out_sync_buffs = (StreamBuffer(stream.out_spec),)
    for out_sync_i, out_sync_buff in zip(out_sync, out_sync_buffs):
        out_sync_buff.feed(out_sync_i, close_input=True)

    ## Get the stream output & compare it
    in_bufs = (StreamBuffer(stream.in_spec),)
    for in_buf, arg in zip(in_bufs, inputs):
        in_buf.feed(arg, close_input=True)

    step_size_iter = iter(itertools.cycle(in_step_sizes))
    i = 0
    while not stream.output_closed:
        step_size = next(step_size_iter)
        inputs_i = [in_buf.read(step_size) for in_buf in in_bufs]

        # FIXME: in_bufs[0]
        try:
            out_stream = stream(*inputs_i, is_last_input=in_bufs[0].output_closed)
        except NotEnoughInputError:
            continue

        # TODO: validate using spec

        # FIXME
        out_stream = out_stream if isinstance(out_stream, tuple) else (out_stream,)
        assert len(out_stream) == 1, "Stream output count mismatch"

        # Ensure the outputs are close
        for out_sync_buff_i, out_stream_i in zip(out_sync_buffs, out_stream):
            stream_out_size = out_sync_buff_i.spec.get_seq_size(out_stream_i)
            out_sync_i = out_sync_buff_i.read(stream_out_size)

            assert out_sync_i.shape == out_stream_i.shape, (
                f"Shape mismatch on step {i} (got {out_stream_i.shape}, expected {out_sync_i.shape})"
            )
            diff = np.abs(out_sync_i - out_stream_i)
            max_error = diff.max()
            assert max_error <= atol, f"Error too large on step {i} (got {max_error}, expected <= {atol})\n{diff}"

        i += 1

    for out_sync_buff_i in out_sync_buffs:
        assert out_sync_buff_i.output_closed, f"Stream output is too short, {out_sync_buff_i.size} outputs remain"
