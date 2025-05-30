import itertools
import logging
import math
import time
from functools import partial
from typing import Callable, Iterable, List, Optional, Tuple

import torch
from z3 import And, Bool, If, Int, Ints, Not, Or, Solver, sat

from torchstream.sequence.seq_spec import SeqSpec
from torchstream.sequence.sequence import Sequence
from torchstream.sliding_window.nan_trick import check_nan_trick, get_nan_map, run_nan_trick
from torchstream.sliding_window.sliding_window_params import SlidingWindowParams
from torchstream.sliding_window.sliding_window_stream import SlidingWindowStream
from torchstream.stream_equivalence import test_stream_equivalent

logger = logging.getLogger(__name__)


class NoSolutionError(Exception):
    pass


class SlidingWindowParamsSolver:
    """
    TODO: doc
    TODO: sort these out
    - The input size to the number of windows is a deterministic, stepwise monotonic linear function.

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
        # Output trimming. See SlidingWindowParams for details.
        self.t_o = Int("t_o")
        self.solver.add(0 <= self.t_o, self.t_o < self.k_o)

    def add_in_out_range_map(
        self,
        in_len: int,
        out_len: int,
        in_nan_range: Tuple[int, int] | None,
        out_nan_range: Tuple[int, int] | None,
    ):
        """
        TODO: doc
        """
        if in_len < 1:
            raise ValueError("The input length must be a strictly positive integer")
        if out_len < 0:
            raise ValueError("The output length must be a non-negative integer")
        if in_nan_range:
            in_nan_range = (int(in_nan_range[0]), int(in_nan_range[1]))
            if not (0 <= in_nan_range[0] < in_nan_range[1] <= in_len):
                raise ValueError("Input range must be non-empty and contained within (0, in_len)")
        if out_nan_range:
            out_nan_range = (int(out_nan_range[0]), int(out_nan_range[1]))
            if not (0 <= out_nan_range[0] < out_nan_range[1] <= out_len):
                raise ValueError("Output range must be non-empty and contained within (0, out_len), or be None")

        # Model the input to output size relation with the number of windows
        constraint_idx = len(self.solver.assertions())
        c = Int(f"c_{constraint_idx}")
        padded_in_len = self.p_l + in_len + self.p_r
        self.solver.add(
            c == If(padded_in_len >= self.k_i, (padded_in_len - self.k_i) / self.s_i + 1, 0),
            out_len == If(c > 0, (c - 1) * self.s_o + self.k_o - 2 * self.t_o, 0),
        )

        # Nan trick - it has many edge cases:
        #   - Input kernels may have gaps (e.g. dilation) and thus not scan all of their inputs
        #   - Output kernels may also have gaps and thus yield disparate outputs
        #   - Nans in the input may be entirely missed because they're past the last window
        #   - Nans in the output may be entirely suppressed due to output trimming
        # As a result, the assumptions we can make are limited:
        #   - If there are multiple non-contiguous regions of nans in the input, we can't determine with certainty
        #     which region of the output results from which region of the input.
        #   - Only the first and last index of the input and output windows are guaranteed to carry over the nans, but
        #     they still may be suppressed by output trimming.
        # FIXME? Add assertions when we have no output
        if not in_nan_range or not out_nan_range:
            return

        # TODO? I could strengthen the constraints on crs/cre by putting the out nan range into a var that could also
        # be the indices trimmed by out_trim. I just need to ensure this is compatible with kernels that have gaps.

        # The window(s) that output nans must necessarily have seen a nan in their input. We'll model this.
        crs, cre = Ints(f"c_{constraint_idx}_rs c_{constraint_idx}_re")
        self.solver.add(
            crs <= cre,
            # crs is the index of the first window that could possibly output the first nan in the output (we have no
            # guarantee that it is indeed that window, this is a lower bound).
            crs >= 0,
            crs >= (out_nan_range[0] - self.k_o + 1 + self.t_o) / self.s_o,
            # Likewise, cre is the index of the last window that could possibly have output the last nan in the output.
            cre < c,
            cre <= (out_nan_range[1] + self.t_o) / self.s_o,
            # [crs, cre] defines a range of windows which necessarily overlaps the input nans. We have no guarantee
            # it fully contains them due to the edge cases listed above.
            self.p_l + in_nan_range[0] < cre * self.s_i + self.k_i,
            self.p_l + in_nan_range[1] >= crs * self.s_i,
        )

        # if not in_nan_range:
        #     return
        # padded_nan_start, padded_nan_stop = self.p_l + in_nan_range[0], self.p_l + in_nan_range[1]
        # if out_nan_range:
        #     # The start of both the input and the output range correspond to the same window. The same can be said
        #     # for the end of the ranges.
        #     # FIXME: notation difference: c above is the number of windows, cs and ce are window indices
        #     crs, cre = Ints(f"c_{constraint_idx}_rs c_{constraint_idx}_re")
        #     self.solver.add(0 <= crs, crs <= cre, cre < c)

        #     self.solver.add(out_nan_range[0] == crs * self.s_o - self.t_o)
        #     self.solver.add(
        #         crs == (If(padded_nan_start >= self.k_i, padded_nan_start - self.k_i + 1, 0) + self.s_i - 1) / self.s_i
        #     )

        #     self.solver.add(out_nan_range[1] == cre * self.s_o + self.k_o - self.t_o)
        #     self.solver.add(cre == If(padded_nan_stop > (c - 1) * self.s_i, c - 1, (padded_nan_stop - 1) / self.s_i))
        # else:
        #     # When there's an output but no in->out range, it necessarily means that the input range was dropped
        #     # due to windows not lining up. Therefore, the input range is fully contained after the last window.
        #     # FIXME!! not the case with output trimming
        #     self.solver.add(Implies(c > 0, padded_nan_start >= self.k_i + self.s_i * (c - 1)))

    def get_violations(self, solution: SlidingWindowParams):
        # TODO: doc
        unsat_solver = Solver()

        trackers = []
        for idx, assertion in enumerate(self.solver.assertions()):
            bool_tracker = Bool(f"assertion_{idx}")
            unsat_solver.assert_and_track(assertion, bool_tracker)
            trackers.append((bool_tracker, assertion))

        unsat_solver.add(
            And(
                self.k_i == solution.kernel_size_in,
                self.s_i == solution.stride_in,
                self.p_l == solution.left_pad,
                self.p_r == solution.right_pad,
                self.k_o == solution.kernel_size_out,
                self.s_o == solution.stride_out,
                self.t_o == solution.out_trim,
            )
        )

        unsat_solver.check()
        violations = [
            expression for (bool_tracker, expression) in trackers if bool_tracker in unsat_solver.unsat_core()
        ]
        return violations

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
                    self.t_o == known_sol.out_trim,
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
                    out_trim=model[self.t_o].as_long(),
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
                        self.t_o != model[self.t_o],
                    )
                )

        return solutions


def _get_infogain(category_counts: Iterable[int]) -> float:
    category_counts = list(category_counts)
    infogain = math.log(sum(category_counts))
    for category_count in category_counts:
        infogain -= (category_count * math.log(category_count)) / sum(category_counts)
    return infogain


def _get_infogain_for_hypotheses(hypotheses: List[SlidingWindowParams], input_size: int, in_nan_idx: Iterable[int]):
    # FIXME!!
    if in_nan_idx:
        in_nan_idx = next(iter(in_nan_idx))
        nan_range = (in_nan_idx, in_nan_idx + 1)
    else:
        nan_range = None
    nan_maps = [get_nan_map(hyp, input_size, nan_range) for hyp in hypotheses]

    groups = [tuple(map_idx) for _, map_idx in itertools.groupby(range(len(nan_maps)), key=lambda i: len(nan_maps[i]))]
    max_out_len = max(len(nan_map) for nan_map in nan_maps)

    # FIXME!!
    if len(groups) > 1:
        return 1.0

    for out_idx in range(max_out_len):
        # new_groups = []
        # for group in groups:
        #     if len(group) == 1:
        #         new_groups.append(group)
        #         continue

        nan_values = [(nan_map[out_idx] if len(nan_map) > out_idx else None) for nan_map in nan_maps]

        unique_values = set(nan_values)
        if 1 in unique_values:
            unique_values.remove(1)
        if len(unique_values) > 1:
            return 1.0

    return 0.0


def find_nan_trick_params_by_infogain(hypotheses: List[SlidingWindowParams]):
    min_in_size = min(hyp.get_min_input_size() for hyp in hypotheses)
    # FIXME!!
    max_in_size = min_in_size + len(hypotheses) + 100

    best_infogain = 0.0
    # FIXME!!
    best_in_size = 100
    for in_size in range(min_in_size, max_in_size + 1):
        infogain = _get_infogain_for_hypotheses(hypotheses, in_size, set())
        if infogain > best_infogain:
            best_infogain = infogain
            best_in_size = in_size
            # print(in_size, infogain, outcomes_count)

    best_nan_idx = None
    for nan_idx in range(0, best_in_size):
        infogain = _get_infogain_for_hypotheses(hypotheses, best_in_size, {nan_idx})
        if infogain > best_infogain:
            best_infogain = infogain
            best_nan_idx = nan_idx
            # print(best_in_size, nan_idx, infogain, outcomes_count)

    # FIXME!!
    if best_infogain == 0.0:
        return None, None

    return best_in_size, best_nan_idx


def _separate_incompatible_hypotheses(hypotheses, history):
    hypotheses = list(hypotheses)

    incompat_hypotheses = []
    for hypothesis in list(hypotheses):
        for record in history:
            if not check_nan_trick(
                hypothesis,
                record["in_seq"].size,
                record["out_seq"].size,
                record["in_nan_range"],
                record["out_nan_idx"],
            ):
                incompat_hypotheses.append(hypothesis)
                hypotheses.remove(hypothesis)
                break

    return hypotheses, incompat_hypotheses


# TODO: allow transforms with multiple sequential inputs
#   -> Or simply call the function multiple times? unsure
@torch.no_grad()
def find_sliding_window_params_for_transform(
    trsfm: Callable,
    input_provider: Callable[[int], Sequence] | SeqSpec,
    out_spec: Optional[SeqSpec] = None,
    init_seq_size: int = 30,
    max_in_seq_size: int = 10_000,
    max_hypotheses_per_step: int = 100,
    atol: float = 1e-5,
) -> List[SlidingWindowParams]:
    """
    Given a sequence-to-sequence transform (neural net, single layer, time series analysis function, ...),
    this function will empirically determine sliding window parameters that correspond to the transform. That is, if
    the transform can be decomposed into a sliding window and a kernel applied to each window. This allows for
    deriving metrics for the transform (time to first output, latency, context size, chunk size needed for
    streaming, ...) as well as making the transform trivially streamable without approximations.

    This is only possible if the transform can be assimilated to a sliding window operation TODO: describe properties
    of this operation

    TODO: rewrite docs

    TODO: handle multi-input/output
    :param input_spec: specification for the input format of the transform. The transform must accept the data format
    described in the input spec as positional arguments.
    :param output_spec: same as input_spec but for the output of the transform. If the transform has multiple
    sequential outputs, they must be returned as an iterable matching the output spec. If the output spec is
    identical to the input spec, it can be omitted, and the input spec will be used instead.
    :param input_provider: a function that takes an integer representing the sequence size, and returns a Sequence of
    this size.
    :param max_hypotheses_per_step: the solver finds up to this amount of sliding windows parameters compatible with
    observations made over the execution of this function. Increasing this value will decrease the number of executions
    of the transforms, but increase the execution time of the solver. If necessary, tune it according to your model's
    performances.
    """
    if max_hypotheses_per_step <= 1:
        raise ValueError("max_hypotheses_per_step must be greater than 1")
    if isinstance(input_provider, SeqSpec):
        in_spec = input_provider
        input_provider = partial(Sequence.randn, in_spec)

    solver = SlidingWindowParamsSolver()
    solver_converged = False
    hypotheses = []
    history = []
    while len(hypotheses) > 1 or not solver_converged:
        # Determine an input size and an input nan range
        if not hypotheses:
            # In the absence of input/output information, use sane defaults
            seq_size = init_seq_size
            in_nan_range = (seq_size // 2, seq_size // 2 + 1)
        else:
            # Once we have a couple of hypotheses, we'll determine our nan trick parameters based on them
            # Get the nan trick parameters that will be the most discriminative of the hypotheses
            seq_size, in_nan_idx = find_nan_trick_params_by_infogain(hypotheses)
            if seq_size is None:
                logger.debug(
                    f"Got {len(hypotheses)} compatible hypotheses from the nan trick, moving on to equivalency "
                    f"checking. Hypotheses:\n{hypotheses}"
                )
                break
            in_nan_range = (in_nan_idx, in_nan_idx + 1) if in_nan_idx is not None else None

        # Verify we're not using the same input parameters twice, which would be a waste of compute
        assert not any(
            record["in_seq"].size == seq_size and record["in_nan_range"] == in_nan_range for record in history
        ), f"Internal error: input parameters ({seq_size}, {in_nan_range}) have already been used"

        # Get an input of said size
        in_seq = input_provider(seq_size)
        if not isinstance(in_seq, Sequence):
            raise TypeError(
                f"The input_provider function {input_provider} returned a {type(in_seq)} when a Sequence was expected"
            )

        # Perform the nan trick on the actual transform
        out_seq, out_nan_idx = run_nan_trick(trsfm, in_seq, in_nan_range, out_spec=(out_spec or in_seq.spec))

        # Keep track of the solver's history
        record = {
            "step": len(history) + 1,
            "in_seq": in_seq.copy(),
            "in_nan_range": in_nan_range,
            "out_seq": out_seq.copy(),
            "out_nan_idx": out_nan_idx,
        }
        history.append(record)

        # Specifically handle transforms that need larger input sizes
        if out_seq.size == 0:
            # TODO: better messages
            if seq_size == max_in_seq_size:
                raise RuntimeError()
            if not hypotheses:
                logger.info(
                    f"Transform failed with input size {seq_size}. Increasing init sequence size to {init_seq_size}"
                )
                init_seq_size = min(10 * init_seq_size, max_in_seq_size)

        # Provide the nan trick results to the solver
        out_nan_range = (out_nan_idx[0], out_nan_idx[-1] + 1) if len(out_nan_idx) else None
        solver.add_in_out_range_map(seq_size, out_seq.size, in_nan_range, out_nan_range)

        # # FIXME!!
        # sol = SlidingWindowParams(kernel_size_in=3, stride_in=2)
        # print("====")
        # print(sol)
        # v = True
        # for violation in solver.get_violations(sol):
        #     v = False
        #     print(violation)
        #     print("----")
        # print("====")
        # if not v:
        #     quit()

        # Eliminate incompatible hypotheses based on these results
        n_hyps_init = len(hypotheses)
        hypotheses, record["rejected_hypotheses"] = _separate_incompatible_hypotheses(hypotheses, history[-1:])
        assert not n_hyps_init or len(hypotheses) < n_hyps_init, "Internal error: no hypotheses were removed"

        # Get new hypotheses
        # TODO: doc
        if not hypotheses and len(out_nan_idx) == out_seq.size:
            if seq_size == max_in_seq_size:
                # TODO: offer a course of action
                logger.warning(
                    f"Your transform outputs NaNs covering the entire output (size={out_seq.size}) given the "
                    f"maximum input size (={seq_size}). This likely means that an operation in your transform "
                    f"broadcasts an input element to all output elements, like a mean, batchnorm, etc... We can't "
                    f"determine sliding window parameters nor stream exactly these types of transforms as their kernel "
                    f"size is technically infinite."
                )
                break
            init_seq_size = min(10 * init_seq_size, max_in_seq_size)
        elif not solver_converged:
            solver_start_time = time.perf_counter()
            hypotheses = solver.get_solutions(hypotheses, max_solutions=max_hypotheses_per_step)
            record["solver_time"] = time.perf_counter() - solver_start_time
            if len(hypotheses) < max_hypotheses_per_step:
                solver_converged = True

            # The solver is not as strict as it should be, so we need to filter out incompatible hypotheses based
            # on past observations.
            hypotheses, _ = _separate_incompatible_hypotheses(hypotheses, history)

        record["hypotheses"] = list(hypotheses)
        record["solver_converged"] = solver_converged
        logger.info(
            f"Step {len(history)}: got {len(hypotheses)} hypothes{'es' if len(hypotheses) != 1 else 'is'} "
            + (f"(max is {max_hypotheses_per_step}) " if len(hypotheses) == max_hypotheses_per_step else "")
            + (f"(solver ran in {record['solver_time'] * 1000:.0f}ms)" if "solver_time" in record else "")
        )

    # FIXME!!
    in_seq = input_provider(100)
    for hypothesis in list(hypotheses):
        try:
            test_stream_equivalent(
                trsfm, SlidingWindowStream(trsfm, hypothesis, in_seq.spec, out_spec), in_seq, atol=atol
            )
        except AssertionError:
            hypotheses.remove(hypothesis)
    if len(hypotheses) > 1:
        logger.info(f"Found {len(hypotheses)} compatible hypotheses:\n{hypotheses}")

    return hypotheses
