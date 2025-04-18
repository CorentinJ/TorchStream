import numpy as np

from torchstream.sliding_window.sliding_window_params import SlidingWindowParams


class DummySlidingWindowTransform:
    def __init__(self, params: SlidingWindowParams):
        self.params = params

    def __call__(self, x: np.ndarray):
        (left_pad, right_pad), num_wins, out_size = self.params.get_metrics_for_input(x.shape[-1])

        x = np.concatenate([np.zeros(left_pad), x])
        if right_pad < 0:
            x = x[:right_pad]
        else:
            x = np.concatenate([x, np.zeros(right_pad)])

        out = np.zeros(out_size)
        for i in range(num_wins):
            in_sli = slice(i * self.params.stride_in, i * self.params.stride_in + self.params.kernel_size_in)
            out_sli = slice(i * self.params.stride_out, i * self.params.stride_out + self.params.kernel_size_out)
            out[out_sli] += np.mean(x[in_sli])

        return out
