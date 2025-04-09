import logging
import math
from typing import Callable, Iterable, List, Optional, Tuple

import numpy as np
import torch
from z3 import And, If, Implies, Int, Ints, Not, Or, Solver, sat

from torchstream.sequence_spec import SeqSpec, Sequence
from torchstream.sliding_window.nan_trick import run_nan_trick
from torchstream.sliding_window.sliding_window_params import SlidingWindowParams

logger = logging.getLogger(__name__)


class NoSolutionError(Exception):
    pass


class SlidingWindowParamsSolver:
    """
    TODO: doc
    TODO: sort these out
    - The input size to the number of windows is a deterministic, stepwise monotonic linear function.
    - On the nan trick: if there's no nan in the output, it necessarily means that the effective input size was smaller
    than the padded input size.

    TODO: padding
    - The most common padding is constant, with zeroes. But there are other padding methods such as reflect or
    circular that sample from the input array to determine the padding values. How does this compromise results from
    the nan trick? Does the transform still qualify as a sliding window approach (e.g. if it reads in the middle of
    the vector to get padding values)?
    """

    def __init__(self):
        # NOTE: possible constructor parameters:
        # - Left padding variable

        # Define the parameters we're trying to uniquely determine
        self.solver = Solver()
        # k_i and s_i are respectively the input kernel size and stride (NOTE: it's technically the kernel span,
        # i.e. the whole span of the kernel if dilation > 1)
        self.k_i, self.s_i = Ints("k_i s_i")
        # It would be highly unusual to have a stride larger than the kernel size, leading to inputs being dropped.
        self.solver.add(self.k_i >= self.s_i, self.s_i > 0)
        # k_o and s_o are respectively the output kernel size and stride. These are both 1 for normal convolutions,
        # but not for transposed convolutions.
        self.k_o, self.s_o = Ints("k_o s_o")
        # Again, it would be strange to have a stride larger than the kernel size, leading to gaps in the output.
        self.solver.add(self.k_o >= self.s_o, self.s_o > 0)
        # The left input padding. I have not yet seen a case where varying the left padding is useful, so we'll
        # assume it constant. Also, there is no point in making the padding higher than the kernel size, as it would
        # waste compute on constant values.
        self.p_l = Int("p_l")
        self.solver.add(0 <= self.p_l, self.p_l < self.k_i)
        # TODO doc
        self.p_r = Int("p_r")
        self.solver.add(0 <= self.p_r, self.p_r < self.k_i)

        # Number of sliding windows for constraints
        # TODO?: remove
        self.cs = []

    def add_in_out_range_map(
        self,
        in_out_len: Tuple[int, int],
        in_out_ranges: Iterable[Tuple[Tuple[int, int], Tuple[int, int] | None]],
    ):
        """
        TODO: doc
        """
        in_len, out_len = int(in_out_len[0]), int(in_out_len[1])
        if in_len < 1:
            raise ValueError("The input length must be a strictly positive integer")
        if out_len < 0:
            raise ValueError("The output length must be a non-negative integer")

        # Input to output size relation with the number of windows
        constraint_idx = len(self.cs)
        c = Int(f"c_{constraint_idx}")
        self.cs.append(c)
        padded_in_len = self.p_l + in_len + self.p_r
        self.solver.add(
            c == If(padded_in_len >= self.k_i, (padded_in_len - self.k_i) / self.s_i + 1, 0),
            out_len == If(c > 0, (c - 1) * self.s_o + self.k_o, 0),
        )

        for range_idx, (in_range, out_range) in enumerate(in_out_ranges):
            in_range = (int(in_range[0]), int(in_range[1]))
            if not (0 <= in_range[0] < in_range[1] <= in_len):
                raise ValueError("Input ranges must be non-empty and contained within (0, in_len)")

            if out_range is not None:
                out_range = (int(out_range[0]), int(out_range[1]))
                if not (0 <= out_range[0] < out_range[1] <= out_len):
                    raise ValueError("Output ranges must be non-empty and contained within (0, out_len), or be None")

            if out_range:
                # The start of both the input and the output range correspond to the same window. The same can be said
                # for the end of the ranges.
                # FIXME: notation difference: c above is the number of windows, cs and ce are window indices
                crs, cre = Ints(f"c_{constraint_idx}_rs{range_idx} c_{constraint_idx}_re{range_idx}")
                self.solver.add(0 <= crs, crs <= cre, cre < c)

                self.solver.add(out_range[0] == crs * self.s_o)
                self.solver.add(
                    crs
                    == (If(self.p_l + in_range[0] >= self.k_i, self.p_l + in_range[0] - self.k_i + 1, 0) + self.s_i - 1)
                    / self.s_i
                )

                self.solver.add(out_range[1] == cre * self.s_o + self.k_o)
                self.solver.add(
                    cre
                    == If(self.p_l + in_range[1] > (c - 1) * self.s_i, c - 1, (self.p_l + in_range[1] - 1) / self.s_i)
                )
            else:
                # When there's an output but no in->out range, it necessarily means that the input range was dropped
                # due to windows not lining up. Therefore, the input range is fully contained after the last window.
                self.solver.add(Implies(c > 0, self.p_l + in_range[0] >= self.k_i + self.s_i * (c - 1)))

    def is_compatible(self, solution: SlidingWindowParams) -> bool:
        with self.solver as temp_solver:
            temp_solver.add(
                self.k_i == solution.kernel_size_in,
                self.s_i == solution.stride_in,
                self.p_l == solution.left_pad,
                self.p_r == solution.right_pad,
                self.k_o == solution.kernel_size_out,
                self.s_o == solution.stride_out,
            )
            return temp_solver.check() == sat

    def get_solutions(
        self, known_solutions: List[SlidingWindowParams] | None = None, max_solutions: int = 100
    ) -> List[SlidingWindowParams]:
        # TODO! doc
        # TODO! a small gain in performance is possible by only iterating on solutions never computed before, and
        # letting the caller handle manage bookkeeping of successful previous solutions. But that require a change
        # in the API and it's not an obvious approach.

        # This context manager will contain the new constraints to the context; they won't be kept upon exiting the
        # context.
        with self.solver as temp_solver:
            solutions = list(known_solutions or [])

            # Verify that the known solutions provided by the caller satisfy the constraints
            for known_sol in solutions:
                known_sol_constraint = And(
                    self.k_i == known_sol.kernel_size_in,
                    self.s_i == known_sol.stride_in,
                    self.p_l == known_sol.left_pad,
                    self.p_r == known_sol.right_pad,
                    self.k_o == known_sol.kernel_size_out,
                    self.s_o == known_sol.stride_out,
                )

                # Exclude that solution from the new solutions
                temp_solver.add(Not(known_sol_constraint))

            # Find new solutions
            while len(solutions) < max_solutions and temp_solver.check() == sat:
                model = temp_solver.model()

                params = SlidingWindowParams(
                    kernel_size_in=model[self.k_i].as_long(),
                    stride_in=model[self.s_i].as_long(),
                    left_pad=model[self.p_l].as_long(),
                    right_pad=model[self.p_r].as_long(),
                    kernel_size_out=model[self.k_o].as_long(),
                    stride_out=model[self.s_o].as_long(),
                )
                solutions.append(params)

                temp_solver.add(
                    Or(
                        self.k_i != model[self.k_i],
                        self.s_i != model[self.s_i],
                        self.p_l != model[self.p_l],
                        self.p_r != model[self.p_r],
                        self.k_o != model[self.k_o],
                        self.s_o != model[self.s_o],
                    )
                )

        return solutions


def _get_infogain(category_counts: Iterable[int]) -> float:
    category_counts = list(category_counts)
    infogain = math.log(sum(category_counts))
    for category_count in category_counts:
        infogain -= (category_count * math.log(category_count)) / sum(category_counts)
    return infogain


def find_nan_trick_params_by_infogain(
    hypotheses: List[SlidingWindowParams],
    min_in_seq_size: int = 1,
    max_in_seq_size: int | None = None,
    in_nan_size: int = 1,
) -> Tuple[int, Tuple[int, int], dict]:
    # TODO: doc
    max_in_seq_size = max_in_seq_size or int(1e100)
    if min_in_seq_size < 1:
        raise ValueError("The minimum sequence size must be a strictly positive integer")
    if max_in_seq_size < min_in_seq_size:
        raise ValueError("The maximum sequence size must be None or greater than the minimum sequence size")

    # Take a subset of the sequence size space
    # TODO: are better bounds possible?
    min_in_seq_size = max(min(sol.get_min_input_size() for sol in hypotheses), min_in_seq_size, in_nan_size)
    max_in_seq_size = min(min_in_seq_size + max(sol.kernel_size_in for sol in hypotheses), max_in_seq_size)

    best_infogain = 0.0
    best_hyps_by_outcome = None
    best_parameters = (None, None)
    for in_seq_size in range(min_in_seq_size, max_in_seq_size + 1):
        # Group hypotheses by their inverse map
        inv_maps_idx = {}
        hyps_by_inv_map = []
        for hypothesis in hypotheses:
            inv_map = hypothesis.get_inverse_map(in_seq_size)

            # FIXME
            inv_map = np.maximum(inv_map, 0)
            inv_map = np.minimum(inv_map, in_seq_size)

            key = inv_map.tobytes()
            if key not in inv_maps_idx:
                inv_maps_idx[key] = len(hyps_by_inv_map)
                hyps_by_inv_map.append((inv_map, []))
            hyps_by_inv_map[inv_maps_idx[key]][1].append(hypothesis)

        if len(hyps_by_inv_map) == 1:
            continue

        # TODO: sublinear algo
        for in_nan_idx in range(in_seq_size - in_nan_size + 1):
            # TODO: algo with varying nan ranges?
            in_nan_range = (in_nan_idx, in_nan_idx + in_nan_size)

            hyps_by_outcome = {}
            for inv_map, hyp_group in hyps_by_inv_map:
                out_nan_range = (
                    np.searchsorted(inv_map[:, 1], in_nan_range[0], side="right"),
                    np.searchsorted(inv_map[:, 0], in_nan_range[1], side="left"),
                )
                if out_nan_range[1] <= out_nan_range[0]:
                    out_nan_range = None
                hyps_by_outcome.setdefault((len(inv_map), out_nan_range), []).extend(hyp_group)

            infogain = _get_infogain(map(len, hyps_by_outcome.values()))
            if infogain > best_infogain:
                best_infogain = infogain
                best_hyps_by_outcome = hyps_by_outcome
                best_parameters = (in_seq_size, in_nan_range)

            # Break early if the solution is optimal
            if len(hyps_by_outcome) == len(hypotheses):
                return *best_parameters, best_hyps_by_outcome

    return *best_parameters, best_hyps_by_outcome


# TODO: allow transforms with multiple sequential inputs
#   -> Or simply call the function multiple times? unsure
# TODO: handle the case where a model always yields an output even if the input is too small?
# TODO: either add a couple of verification rounds, either handle the verification through a stream
@torch.no_grad()
def find_sliding_window_params_for_transform(
    trsfm: Callable,
    input_spec: SeqSpec,
    output_spec: Optional[SeqSpec] = None,
    input_provider: Optional[Callable[[int], Sequence]] = None,
    min_in_seq_size: int = 1,
    max_in_seq_size: int = 10_000,
    max_hypotheses_per_step: int = 100,
    max_in_kernel_gap: int = 10,
) -> List[SlidingWindowParams]:
    """
    Given a sequence-to-sequence transform (neural net, single layer, time series analysis function, ...),
    this function will empirically determine sliding window parameters that correspond to the transform. That is, if
    the transform can be decomposed into a sliding window and a kernel applied to each window. This allows for
    deriving metrics for the transform (time to first output, latency, context size, chunk size needed for
    streaming, ...) as well as making the transform trivially streamable without approximations.

    This is only possible if the transform can be assimilated to a sliding window operation TODO: describe properties
    of this operation

    TODO: handle multi-input/output
    :param input_spec: specification for the input format of the transform. The transform must accept the data format
    described in the input spec as positional arguments.
    :param output_spec: same as input_spec but for the output of the transform. If the transform has multiple
    sequential outputs, they must be returned as an iterable matching the output spec. If the output spec is
    identical to the input spec, it can be omitted, and the input spec will be used instead.
    :param input_provider: a function that takes an integer representing the sequence size, and returns a sequence of
    this size matching the input spec. By default, a random normal (rounded for int types) is sampled according to
    the input specification.
    :param max_hypotheses_per_step: the solver finds up to this amount of sliding windows parameters compatible with
    observations made over the execution of this function. Increasing this value will decrease the number of executions
    of the transforms, but increase the execution time of the solver. If necessary, tune it according to your model's
    performances.
    :param max_in_kernel_gap: some kernels do not use all the elements of the window to compute their output. For
    instance, dilated convolutions (à trous) have wide kernels but skip some elements of their inputs. This breaks
    assumptions made by the solver, unless the largest possible such gap in the kernel is known in advance. If you
    suspect your model has such gaps, set this parameter to a safe higher bound. For reference, a kernel of size 5
    that returns the sum of the first and last element of the window has a gap of 3. A convolution with dilation
    greater than 1 has gaps of size <dilation - 1>.
    """
    output_spec = output_spec or input_spec

    solver = SlidingWindowParamsSolver()

    # Until we have well-formed pairs of input/output examples from the transform, we'll determine the input
    # parameters based on heuristics.
    in_nan_size = max(max_in_kernel_gap, 1)
    default_seq_size = min(5 * max(min_in_seq_size, in_nan_size), max_in_seq_size)

    step = 1
    hypotheses = []
    prev_nan_trick_params = set()
    while len(hypotheses) != 1:
        # Determine an input size and an input nan range
        if not hypotheses:
            # In the absence of input/output information, use sane defaults
            seq_size = default_seq_size
            in_nan_range = (seq_size // 2 - in_nan_size // 2, seq_size // 2 + (in_nan_size + 1) // 2)
            hyps_by_outcome = None
        else:
            # Once we have a couple of hypotheses, we'll determine our nan trick parameters based on them
            # We might be able to reduce the nan range size if the hypotheses are all compatible with a smaller one
            if len(hypotheses) < max_hypotheses_per_step:
                in_nan_size = np.clip(max(hyp.kernel_size_in for hyp in hypotheses) - 1, 1, max_in_kernel_gap)

            # Get the nan trick parameters that will be the most discriminative of the hypotheses
            seq_size, in_nan_range, hyps_by_outcome = find_nan_trick_params_by_infogain(
                hypotheses, min_in_seq_size, max_in_seq_size, in_nan_size
            )
            if seq_size is None:
                # TODO: is this problematic at all for streaming? It seems equivalent sliding windows parameters would
                # lead to the same parameters for the streaming implementation of the transform.
                logger.info(
                    f"Got {len(hypotheses)} sliding window parameters that are consistent with the transform, but they "
                    f"cannot be discriminated further. All possible parameters will be returned, with the most likely "
                    f"ones first."
                )
                break

        assert (seq_size, in_nan_range) not in prev_nan_trick_params, "Internal error: nan trick parameters repeated"
        prev_nan_trick_params.add((seq_size, in_nan_range))

        # Perform the nan trick on the actual transform
        out_size, out_nan_range = run_nan_trick(trsfm, input_provider, seq_size, in_nan_range)

        if out_size == 0:
            # TODO: better messages
            if seq_size == max_in_seq_size:
                raise RuntimeError()
            min_in_seq_size = seq_size + 1
            logger.info(
                f"Transform failed with input size {seq_size}. Increasing min sequence size to {min_in_seq_size}"
            )
            if not hypotheses:
                default_seq_size = min(10 * default_seq_size, max_in_seq_size)

        # Provide the nan trick results to the solver
        # TODO: change signature
        solver.add_in_out_range_map((seq_size, out_size), [(in_nan_range, out_nan_range)])

        # Track the hypotheses that are compatible with the new results
        compatible_prev_hypotheses = hyps_by_outcome.pop((out_size, out_nan_range), []) if hyps_by_outcome else []
        if hypotheses:
            logger.info(f"{len(hypotheses) - len(compatible_prev_hypotheses)}/{len(hypotheses)} hypotheses eliminated")

        # Get new hypotheses, keeping the previous ones that are compatible with the new results
        # TODO: doc
        if out_nan_range and not hypotheses and out_nan_range[0] == 0 and out_nan_range[1] == out_size:
            if seq_size == max_in_seq_size:
                # TODO: offer a course of action
                logger.warning(
                    f"Your transform outputs NaNs covering the entire output (size={out_size}) given the "
                    f"maximum input size (={seq_size}). This likely means that an operation in your transform "
                    f"broadcasts an input element to all output elements, like a mean, batchnorm, etc... We can't "
                    f"determine sliding window parameters nor stream exactly these types of transforms as their kernel "
                    f"size is technically infinite."
                )
                break
            default_seq_size = min(10 * default_seq_size, max_in_seq_size)
        else:
            hypotheses = solver.get_solutions(compatible_prev_hypotheses, max_solutions=max_hypotheses_per_step)

        # Also verify that the hypotheses that have been eliminated are not in the solver's output
        if hyps_by_outcome:
            eliminated_hyps = [hyp for hyps in hyps_by_outcome.values() for hyp in hyps]
            assert all(hyp not in hypotheses for hyp in eliminated_hyps), (
                "Internal error: some hypotheses that have been eliminated were found in the solver's output"
            )

        logger.info(
            f"Step {step}: got {len(hypotheses)} hypothes{'es' if len(hypotheses) > 1 else 'is'} "
            + (f" (max is {max_hypotheses_per_step})" if len(hypotheses) == max_hypotheses_per_step else "")
        )

        step += 1

    # TODO: handle no solution

    # At this point we may have multiple solutions.
    # TODO: doc
    def sliding_window_params_simplicity_score(sliding_window_params: SlidingWindowParams) -> int:
        if sliding_window_params.stride_in == 1 and sliding_window_params.kernel_size_in == 1:
            return 2
        if sliding_window_params.stride_out == 1 and sliding_window_params.kernel_size_out == 1:
            return 1
        return 0

    hypotheses = sorted(hypotheses, key=sliding_window_params_simplicity_score, reverse=True)

    return hypotheses
