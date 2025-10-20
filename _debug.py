from torchstream.sliding_window.sliding_window_params import SlidingWindowParams
from torchstream.sliding_window.sliding_window_params_solver import _compare_sli_params_str

a = SlidingWindowParams(3, 3, 1)
b = SlidingWindowParams(4, 3, 2)


print(_compare_sli_params_str(a, b))

for sol in [a, b]:
    for k in a.get_inverse_kernel_map(30):
        print(k)
    print("----")
    print()
