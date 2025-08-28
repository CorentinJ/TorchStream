import logging
import random

from z3 import If, Ints, Or, Solver, sat

logger = logging.getLogger(__name__)


def z3max(a, b):
    return If(a > b, a, b)


def z3min(a, b):
    return If(a < b, a, b)


class DEBUGSampler:
    def __init__(self):
        self.optimizer = Solver()
        self.s_i, self.s_o = Ints("s_i s_o")
        self.optimizer.add(self.s_i > 0, self.s_o > 0)
        self.in_size_bias, self.out_size_bias, self.in_delay, self.out_delay, self.in_context_size = Ints(
            "isb osb id od ics"
        )
        self.optimizer.add(
            self.in_size_bias >= 0,
            self.in_size_bias < self.s_i,
            # self.out_size_bias >= 0,
            # self.out_size_bias < self.s_o,
            self.in_delay >= 0,
            self.in_delay < self.s_i,
            # self.out_delay >= 0,
            # self.out_delay < self.s_o,
        )

    def add(self, in_len: int, out_len: int, out_trim: int):
        num_wins = z3max(0, (in_len + self.in_size_bias) / self.s_i)
        out_len_var = z3max(0, (num_wins - 1) * self.s_o + self.out_size_bias)

        last_eff_win_idx = (in_len - self.in_delay) / self.s_i
        out_trim_var = (last_eff_win_idx + 1) * self.s_o - self.out_delay

        self.optimizer.add(out_len == out_len_var, out_trim == out_trim_var)

    def get_new_solutions(self):
        sols = []
        with self.optimizer as temp_solver:
            while len(sols) < 10:
                check = temp_solver.check()

                if check == sat:
                    model = temp_solver.model()
                    model_values = (
                        model[self.s_i].as_long(),
                        model[self.s_o].as_long(),
                        model[self.in_size_bias].as_long(),
                        model[self.out_size_bias].as_long(),
                        model[self.in_delay].as_long(),
                        model[self.out_delay].as_long(),
                    )

                    # Enforce new solutions only
                    new_sol_constraint = Or(
                        self.s_i != model[self.s_i],
                        self.s_o != model[self.s_o],
                        self.in_size_bias != model[self.in_size_bias],
                        self.out_size_bias != model[self.out_size_bias],
                        self.in_delay != model[self.in_delay],
                        self.out_delay != model[self.out_delay],
                    )
                    temp_solver.add(new_sol_constraint)

                    sols.append(model_values)

                else:
                    break

        return sols


# num_wins = max(0, (in_seq.size + self.in_size_bias) // self.stride_in)
# out_size = max(0, (num_wins - 1) * self.stride_out + self.out_size_bias)
# sufficient_input = in_seq.size and num_wins and out_size

# # See where the output should be trimmed
# if in_seq.input_closed:
#     out_trim_end = out_size
# else:
#     last_eff_win_idx = (in_seq.size - self.in_delay) // self.stride_in
#     out_trim_end = min((last_eff_win_idx + 1) * self.stride_out - self.out_delay, out_size)

stride_in, stride_out, in_size_bias, out_size_bias, in_delay, out_delay, ctx = (3, 1, 1, -4, 2, 5, 14)


def fn(in_len):
    num_wins = max(0, (in_len + in_size_bias) // stride_in)
    out_len = max(0, (num_wins - 1) * stride_out + out_size_bias)

    last_eff_win_idx = (in_len - in_delay) // stride_in
    out_trim = (last_eff_win_idx + 1) * stride_out - out_delay

    return out_len, out_trim


sampler = DEBUGSampler()


for in_len in range(20):
    in_len = random.randint(1, 200)
    out_len, out_trim = fn(in_len)
    sampler.add(in_len, out_len, out_trim)
    print((in_len, out_len, out_trim))

    sols = sampler.get_new_solutions()
    print(f"{in_len}: {len(sols)} sols: {sols}")
