import torch

from torchstream.patching.call_intercept import intercept_calls
from torchstream.sequence.seq_spec import SeqSpec


def work_fn():
    a = torch.arange(5)
    b = torch.cumsum(a, dim=0)
    c = b // 3
    d = torch.cumsum(c, dim=0)
    return d - a


if __name__ == "__main__":

    def handler(*args, original_fn, callstack_locs, **kwargs):
        print(f"Intercepted {original_fn} call from {hash(tuple(callstack_locs))}")
        return original_fn(*args, **kwargs)

    with intercept_calls("torch.cumsum", handler, pass_original_fn=True, pass_callstack_locs=True):
        for _ in range(2):
            result = work_fn()
            print("Result:", result)

SeqSpec()
