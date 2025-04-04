from collections.abc import Collection
from typing import Callable

import numpy as np
import pytest
import torch

from tests.rng import set_seed
from torchstream.buffers.stream_buffer import StreamBuffer
from torchstream.stream import Stream


def _to_np(x):
    return x.cpu().numpy() if torch.is_tensor(x) else x


@pytest.mark.skip("Not a test")
@torch.no_grad()
def test_stream_equivalent(stream: Stream, sync_fn: Callable, inputs: tuple, in_step_size=10, atol=1e-5, seed=None):
    """
    Tests if a stream implementation gives outputs close or equivalent to its synchronous counterpart.

    Both the stream and the sync function must take the same arguments, and return the same number of outputs.
    Outputs must be sequential data of the same shape.

    """
    if seed:
        set_seed(seed)

    inputs = inputs if isinstance(inputs, tuple) else (inputs,)

    ## Get the sync output
    out_sync = _to_np(sync_fn(*inputs))
    out_sync = tuple(out_sync) if isinstance(out_sync, Collection) else (out_sync,)
    assert len(out_sync) == len(stream.out_dims), (
        f"The sync function returned {len(out_sync)} outputs, expected {len(stream.out_dims)} outputs"
    )

    ## Get the stream output
    in_bufs = [StreamBuffer(dim) for dim in in_seq_dims]
    out_bufs = [StreamBuffer(dim) for dim in out_seq_dims]

    [in_buf.feed(arg, close_input=True) for in_buf, arg in zip(in_bufs, inputs)]

    while not in_bufs[0].output_closed:
        args_i = [in_buf.read(in_step_size) for in_buf in in_bufs]

        out = stream(*args_i, is_last_input=in_bufs[0].output_closed)

        out = out if isinstance(out, tuple) else (out,)
        assert len(out) == len(out_bufs), "Stream output count mismatch"

        [out_bufs[i].feed(_to_np(out[i])) for i in range(len(out_bufs))]

    # FIXME!!
    # assert stream.input_closed
    out_stream = [out_buf.read() for out_buf in out_bufs]

    # Ensure the outputs are close
    for out_sync_i, out_stream_i in zip(out_sync, out_stream):
        assert out_stream_i.shape == out_sync_i.shape, "Shape mismatch"
        max_error = np.abs(out_sync_i - out_stream_i).max()
        assert max_error <= atol, (max_error, atol)
