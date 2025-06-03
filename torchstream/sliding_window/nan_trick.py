import logging
from typing import Callable, Optional, Tuple

import numpy as np
from z3 import And, Bool, Not, Or, Solver, unsat

from torchstream.sequence.seq_spec import SeqSpec
from torchstream.sequence.sequence import Sequence
from torchstream.sliding_window.sliding_window_params import SlidingWindowParams

logger = logging.getLogger(__name__)


def get_out_nan_idx(x: Sequence) -> np.ndarray:
    # TODO! doc
    # TODO: numpy() function
    # TODO: is_nan -> any reduction instead
    x = x.data.mean(axis=tuple(i for i in range(x.ndim) if i != x.dim))

    return np.where(np.isnan(x))[0]


def run_nan_trick(
    trsfm: Callable,
    in_seq: Sequence,
    in_nan_range: Tuple[int, int] | None,
    out_spec: Optional[SeqSpec] = None,
) -> Tuple[Sequence, np.ndarray]:
    """
    TODO: doc

    TODO: handle multi-input/output
    """
    if not in_seq.size:
        raise ValueError(f"Input sequence size must be greater than 0, got {in_seq.size}")
    if in_nan_range and not (0 <= in_nan_range[0] < in_nan_range[1] <= in_seq.size):
        raise ValueError(f"Nan range must be positive and within the input sequence size, got {in_nan_range}")
    out_spec = out_spec or in_seq.spec

    # Corrupt the given range of the input sequence with NaNs
    if in_nan_range:
        in_seq[slice(*in_nan_range)] = float("nan")

    # Forward the input through the transform
    logger.debug(f"Running transform with input size {in_seq.size} and nans at {in_nan_range}")
    try:
        out_seq = Sequence.apply(trsfm, in_seq, out_spec)
    except RuntimeError as e:
        # We'll assume that RuntimeError are conv errors for a too small input size
        # TODO: more reliable mechanism
        # TODO: handle errors due to nans

        logger.info(f"Transformed failed with {repr(e)}")

        return Sequence.empty(out_spec), np.empty(0, dtype=np.int64)

    out_nan_idx = get_out_nan_idx(out_seq)
    logger.debug(f"Got a {tuple(out_seq.shape)} shaped output with nans at {out_nan_idx}")

    return out_seq, out_nan_idx


# TODO: tests
def determine_kernel_sparsity(
    params: SlidingWindowParams,
    in_len: int,
    in_nan_range: Tuple[int, int],
    out_nan_idx: np.ndarray,
) -> Tuple[np.ndarray | None, np.ndarray | None]:
    # TODO! doc

    _, num_wins, _ = params.get_metrics_for_input(in_len)

    solver = Solver()
    corrupted_wins = [Bool("corrupted_win_" + str(i)) for i in range(num_wins)]
    kernel_in = [Bool("kernel_in_" + str(i)) for i in range(params.kernel_size_in)]
    kernel_out = [Bool("kernel_out_" + str(i)) for i in range(params.kernel_size_out)]

    # Apply the kernel priors
    for idx, val in enumerate(params.kernel_in_sparsity):
        if val == 0:
            solver.add(kernel_in[idx] == False)
        elif val == 2:
            solver.add(kernel_in[idx] == True)
    for idx, val in enumerate(params.kernel_out_sparsity):
        if val == 0:
            solver.add(kernel_out[idx] == False)
        elif val == 2:
            solver.add(kernel_out[idx] == True)

    for win_idx, ((in_start, in_stop), (out_start, out_stop)) in enumerate(params.iter_kernel_map(num_wins)):
        # The kernel can only output nans (=be corrupted) if it has any overlap with the input nans
        if in_nan_range[0] < in_stop and in_start < in_nan_range[1]:
            kernel_in_nan_range = (
                max(in_nan_range[0], in_start) - in_start,
                min(in_nan_range[1], in_stop) - in_start,
            )
            corrupted_wins[win_idx] = Or(*[kernel_in[i] for i in range(*kernel_in_nan_range)])
        else:
            solver.add(corrupted_wins[win_idx] == False)

    for out_idx, inv_map in enumerate(params.get_inverse_kernel_map(in_len)):
        if out_idx in out_nan_idx:
            solver.add(
                Or(
                    *[
                        And(corrupted_wins[win_idx], kernel_out[kernel_out_idx])
                        for win_idx, in_start, _, kernel_out_idx in inv_map
                    ]
                )
            )
        else:
            solver.add(
                And(
                    *[
                        Not(And(corrupted_wins[win_idx], kernel_out[kernel_out_idx]))
                        for win_idx, in_start, _, kernel_out_idx in inv_map
                    ]
                )
            )

    if solver.check() == unsat:
        return None, None

    model = solver.model()
    kernel_in_values = np.zeros(params.kernel_size_in, dtype=np.int64)
    kernel_out_values = np.zeros(params.kernel_size_out, dtype=np.int64)
    for i in range(params.kernel_size_in):
        if model[kernel_in[i]] == True:
            kernel_in_values[i] = 2
        elif model[kernel_in[i]] == False:
            kernel_in_values[i] = 0
        else:
            kernel_in_values[i] = 1
    for i in range(params.kernel_size_out):
        if model[kernel_out[i]] == True:
            kernel_out_values[i] = 2
        elif model[kernel_out[i]] == False:
            kernel_out_values[i] = 0
        else:
            kernel_out_values[i] = 1

    return kernel_in_values, kernel_out_values


def get_nan_map(
    params: SlidingWindowParams,
    in_len: int,
    in_nan_range: Tuple[int, int] | None,
):
    # TODO! doc
    _, num_wins, out_len = params.get_metrics_for_input(in_len)

    nan_map = np.zeros(out_len, dtype=np.int64)
    if not in_nan_range:
        return nan_map

    for (in_start, in_stop), (out_start, out_stop) in params.iter_kernel_map(num_wins):
        # The kernel can output nans only if it has any overlap with the input nans
        if in_nan_range[0] < in_stop and in_start < in_nan_range[1]:
            # The kernel is only GUARANTEED to output nans if its first or last element are nan (otherwise the kernel
            # might have gaps and these gaps might be precisely aligned with the nans).
            guaranteed_nan_output = (
                in_nan_range[0] <= in_start < in_nan_range[1] or in_nan_range[0] < in_stop <= in_nan_range[1]
            )
            # Likewise, the output kernel might have gaps, so we can only guarantee that the first and last elements
            # of the output window are nans (marked with 2)
            if guaranteed_nan_output and 0 <= out_start < out_len:
                nan_map[out_start] = 2
            if guaranteed_nan_output and 0 < out_stop <= out_len:
                nan_map[out_stop - 1] = 2

            # Everywhere else in the output window, we have an unknown as to whether the output is nan or not
            # (marked with 1)
            unknown_sli = slice(max(0, out_start), min(out_stop, out_len))
            nan_map[unknown_sli] = np.maximum(nan_map[unknown_sli], 1)

    return nan_map
