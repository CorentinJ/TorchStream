# TODO: doc
class ThresholdHarvester:
    def __init__(self, lower_bound: int = 0, initial: int = 10):
        assert 0 <= lower_bound <= initial
        self.lower_bound = lower_bound
        self.last_nonempty = None
        self.gallop_value = 1

    def next_p(self) -> int:
        # If we have not found anything yet in the current range, sample at the upper bound of the range
        # so we can gallop on to the next one immediately after if this one turns out to be empty as well
        if self.last_nonempty is None:
            return self.lower_bound + self.gallop_value

        # If we have harvested a value in the current range, sample halfway between that value and the
        # known lower bound so we increase our chances of harvesting the minimum while quickly exploring
        # the range at the same time
        return (self.lower_bound + self.last_nonempty) // 2

    def update(self, result: int):
        p = self.next_p()

        if result is None:
            if self.last_nonempty is None:
                self.gallop_value *= 2

            self.lower_bound = p + 1

            if self.last_nonempty is not None and p >= self.last_nonempty:
                self.last_nonempty = None

        else:
            self.last_nonempty = result
            self.gallop_value = 1


# import random
# from collections import defaultdict
# from random import randint

# random.seed(42)  # For reproducibility


# def run_policy(draw):
#     ctrl = ThresholdHarvester()
#     for _ in range(100):
#         p = ctrl.next_p()
#         x = draw(p)
#         ctrl.update(x)


# stack = defaultdict(int)
# for _ in range(30):
#     stack[randint(0, 20) + 300] += 1
# stack = {k: v for k, v in sorted(stack.items(), key=lambda item: item[0])}


# def draw(p):
#     mink = min(stack or [None])

#     ks = [k for k in stack if k <= p]
#     k = random.choice(ks) if ks else None
#     if k is None:
#         print(f"\x1b[31m Failed to sample with p={p} \x1b[39m", sep="")
#         return None

#     stack[k] -= 1
#     if stack[k] == 0:
#         del stack[k]

#     if k == mink:
#         print(f"\x1b[32m Sampled minimum={k} with p={p} \x1b[39m", sep="")
#     else:
#         print(f"\x1b[31m Sampled {k} with p={p} when minimum={mink} exists \x1b[39m", sep="")

#     return k


# print(stack)
# run_policy(draw)
