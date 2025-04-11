from collections.abc import Collection
import itertools
from typing import Callable, Optional, Tuple

import numpy as np
import pytest
import torch

from tests.rng import set_seed
from torchstream.buffers.stream_buffer import StreamBuffer
from torchstream.sequence_spec import Sequence
from torchstream.stream import Stream


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

    ## Get the sync output
    out_sync = sync_fn(*inputs)
    out_sync = tuple(out_sync) if not isinstance(out_sync, Collection) else (out_sync,)
    assert len(out_sync) == len(stream.out_spec), (
        f"The sync function returned {len(out_sync)} outputs, expected {len(stream.out_spec)} outputs"
    )

    ## Get the stream output
    in_bufs = (StreamBuffer(stream.in_spec),)
    for in_buf, arg in zip(in_bufs, inputs):
        in_buf.feed(arg, close_input=True)

    out_bufs = (StreamBuffer(stream.out_spec),)
    step_size_iter = iter(itertools.cycle(in_step_sizes))
    while not stream.output_closed:
        step_size = next(step_size_iter)
        inputs_i = [in_buf.read(step_size) for in_buf in in_bufs]

        # FIXME: in_bufs[0]
        out = stream(*inputs_i, is_last_input=in_bufs[0].output_closed)

        # FIXME: format
        out = out if isinstance(out, tuple) else (out,)
        assert len(out) == len(out_bufs), "Stream output count mismatch"

        [out_bufs[i].feed(out[i]) for i in range(len(out_bufs))]
    out_stream = [out_buf.read() for out_buf in out_bufs]

    # Ensure the outputs are close
    for out_sync_i, out_stream_i in zip(out_sync, out_stream):
        # FIXME: do this in validation
        assert out_stream_i.shape == out_sync_i.shape, (
            f"Shape mismatch: sync output is {out_sync_i.shape}-shaped, stream output is {out_stream_i.shape}-shaped"
        )
        max_error = np.abs(out_sync_i - out_stream_i).max()
        assert max_error <= atol, (max_error, atol)
