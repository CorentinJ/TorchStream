# TODO: doc
class ThresholdHarvester:
    def __init__(self, lower_bound: int = 0, initial: int = 10):
        assert 0 <= lower_bound <= initial
        self.lower_bound = lower_bound
        self.upper_bound = initial
        self.last_nonempty = None

    def next_p(self):
        if self.last_nonempty is None:
            return self.upper_bound
        if self.last_nonempty >= self.lower_bound:
            return (self.lower_bound + self.last_nonempty) // 2
        else:
            return (self.lower_bound + self.upper_bound) // 2

    def update(self, result):
        p = self.next_p()

        if result is None:
            self.lower_bound = p + 1

            if p == self.upper_bound:
                self.upper_bound *= 2

        else:
            self.last_nonempty = result


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
# for _ in range(5):
#     stack[randint(0, 1000) + 300] += 1
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
