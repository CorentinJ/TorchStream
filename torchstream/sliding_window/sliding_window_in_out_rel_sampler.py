import logging
from typing import List

from z3 import And, If, Ints, Not, Or, Solver, sat

logger = logging.getLogger(__name__)


class SlidingWindowInOutRelSampler:
    def __init__(self):
        # Input and output strides of the sliding window.
        self.s_i, self.s_o = Ints("s_i s_o")
        # Input and output size biases in computation, canonicalized to ensure uniqueness of the relation
        self.isbc, self.osbc = Ints("isbc osbc")

        self.optimizer = Solver()
        self.optimizer.add(
            self.s_i > 0,
            self.s_o > 0,
            self.isbc >= 0,
            self.isbc < self.s_i,
            # osbc is the only parameter that can be negative -> no constraint here
        )

        # Indicates if more solutions are available
        self.exhausted = False

    # FIXME: name
    def add_in_out_size(self, in_len: int, out_len: int):
        """
        TODO: doc
        """
        if in_len < 1:
            raise ValueError("The input length must be a strictly positive integer")
        if out_len < 0:
            raise ValueError("The output length must be a non-negative integer")

        # FIXME! optim
        def z3max(a, b):
            return If(a > b, a, b)

        out_len_t1 = (in_len + self.isbc) / self.s_i
        out_len_var = z3max(0, out_len_t1 * self.s_o + self.osbc)
        self.optimizer.add(out_len == out_len_var)

    def get_new_solutions(self, known_sols: List, max_sols=10):
        # TODO: doc
        out_sols = list(known_sols)

        # Exclude previous solutions from the search
        self.optimizer.push()
        for sol in out_sols:
            self.optimizer.add(
                Not(And(self.s_i == sol[0], self.s_o == sol[1], self.isbc == sol[2], self.osbc == sol[3]))
            )

        # Search for newer solutions
        while len(out_sols) < max_sols:
            if self.optimizer.check() == sat:
                model = self.optimizer.model()
                model_values = (
                    model[self.s_i].as_long(),
                    model[self.s_o].as_long(),
                    model[self.isbc].as_long(),
                    model[self.osbc].as_long(),
                )
                out_sols.append(model_values)

                # Enforce new solutions only
                new_sol_constraint = Or(
                    self.s_i != model[self.s_i],
                    self.s_o != model[self.s_o],
                    self.isbc != model[self.isbc],
                    self.osbc != model[self.osbc],
                )
                self.optimizer.add(new_sol_constraint)
            else:
                self.exhausted = True
                break
        self.optimizer.pop()

        return out_sols
