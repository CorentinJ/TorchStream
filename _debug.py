from z3 import Solver, Bool

unsat_solver = Solver()

trackers = []
for idx, assertion in enumerate(self.solver.assertions()):
    bool_tracker = Bool(f"assertion_{idx}")
    unsat_solver.assert_and_track(assertion, bool_tracker)
    trackers.append((bool_tracker, assertion))

unsat_solver.check()
violations = [
    expression for (bool_tracker, expression) in trackers if bool_tracker in unsat_solver.unsat_core()
]
