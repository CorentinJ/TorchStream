import logging
from typing import Callable, Optional, Tuple, Union

import numpy as np
import torch

from torchstream.sequence.seq_spec import SeqSpec
from torchstream.sequence.sequence import Sequence
from torchstream.sliding_window.dummy_sliding_window_transform import DummySlidingWindowTransform
from torchstream.sliding_window.sliding_window_params import SlidingWindowParams

logger = logging.getLogger(__name__)


def set_nan_range(
    x: Sequence,
    range: Union[slice, Tuple[int, int]],
    dim: int = -1,
) -> Sequence:
    # TODO! doc
    if not isinstance(range, slice):
        range = slice(*range)

    slices = [slice(None)] * x.ndim
    slices[dim] = range

    x[tuple(slices)] = float("nan")


def get_nan_range(x: Sequence, dim: int = -1) -> Tuple[int, int] | None:
    # TODO! doc
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()

    dim = range(x.ndim)[dim]
    x = x.mean(axis=tuple(i for i in range(x.ndim) if i != dim))

    corrupted_idx = np.where(np.isnan(x))[0]

    if not len(corrupted_idx):
        return None
    return corrupted_idx[0], corrupted_idx[-1] + 1


def run_nan_trick(
    trsfm: Callable,
    in_seq_size: int,
    in_nan_range: Tuple[int, int],
    input_spec: SeqSpec,
    output_spec: Optional[SeqSpec] = None,
    # TODO!: much easier to provide a Sequence instead of seq size + spec + provider?
    input_provider: Optional[Callable[[int], Sequence]] = None,
) -> Tuple[int, Tuple[int, int] | None]:
    """
    TODO: doc

    TODO: handle multi-input/output
    :param input_spec: specification for the input format of the transform. The transform must accept the data format
    described in the input spec as positional arguments.
    :param output_spec: same as input_spec but for the output of the transform. If the transform has multiple
    sequential outputs, they must be returned as an iterable matching the output spec. If the output spec is
    identical to the input spec, it can be omitted, and the input spec will be used instead.
    :param input_provider: a function that takes an integer representing the sequence size, and returns a sequence of
    this size matching the input spec. By default, a random normal (rounded for int types) is sampled according to
    the input specification.
    """
    if in_seq_size < 1:
        raise ValueError(f"Input sequence size must be greater than 0, got {in_seq_size}")
    if not (0 <= in_nan_range[0] < in_nan_range[1] <= in_seq_size):
        raise ValueError(f"Nan range must be positive and within the input sequence size, got {in_nan_range}")
    output_spec = output_spec or input_spec
    input_provider = input_provider or input_spec.randn

    # Get the input sequence of given size
    x = input_provider(in_seq_size)
    if not input_spec.get_seq_size(x) == in_seq_size:
        raise RuntimeError(
            f"Input provided by {input_provider} was expected to have sequence size {in_seq_size}, got {x.shape}"
        )

    # Corrupt the given range of the input sequence with NaNs
    set_nan_range(x, in_nan_range, dim=input_spec.seq_dim)

    # Forward the input through the transform
    logger.debug(f"Running transform with input size {in_seq_size} and nans at {in_nan_range}")
    try:
        # FIXME: output format
        y = trsfm(x)
    except RuntimeError as e:
        # We'll assume that RuntimeError are conv errors for a too small input size
        # TODO: more reliable mechanism
        # TODO: handle errors due to nans

        logger.info(f"Transformed failed with {repr(e)}")

        return 0, None

    out_size = output_spec.get_seq_size(y)
    out_nan_range = get_nan_range(y, dim=output_spec.seq_dim)
    logger.debug(f"Got a {tuple(y.shape)} shaped output with nans at {out_nan_range}")

    return out_size, out_nan_range


def check_nan_trick(
    sliding_window_params: SlidingWindowParams,
    in_len: int,
    out_len: int,
    nan_in_range: Tuple[int, int],
    nan_out_range: Tuple[int, int] | None,
):
    # TODO! doc
    tsfm = DummySlidingWindowTransform(sliding_window_params)

    x = np.random.randn(in_len)
    set_nan_range(x, nan_in_range)

    out = tsfm(x)
    if len(out) != out_len:
        return False, f"expected out len {out_len}, got {len(out)}"

    actual_nan_out_range = get_nan_range(out)
    if actual_nan_out_range != nan_out_range:
        return False, f"expected out range {nan_out_range}, got {actual_nan_out_range}"

    return True, None
