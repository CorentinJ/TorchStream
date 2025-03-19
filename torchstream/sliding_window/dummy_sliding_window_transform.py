import numpy as np

from torchstream.sliding_window.sliding_window_params import SlidingWindowParams


class DummySlidingWindowTransform:
    def __init__(self, params: SlidingWindowParams):
        self.params = params

    def __call__(self, x: np.ndarray):
        # FIXME!
        right_pad = self.params.left_pad

        x = np.concatenate([np.zeros(self.params.left_pad), x])
        if right_pad < 0:
            x = x[:right_pad]
        else:
            x = np.concatenate([x, np.zeros(right_pad)])

        # TODO: make methods of sliding window params
        # num_windows = (len(x) - self.params.kernel_size_in) / self.params.stride_in + 1
        # FIXME!
        # assert num_windows.is_integer()
        num_windows = (len(x) - self.params.kernel_size_in) // self.params.stride_in + 1
        num_windows = int(num_windows)
        output_length = (num_windows - 1) * self.params.stride_out + self.params.kernel_size_out

        out = np.zeros((output_length, 2), dtype=np.int64)
        out[:, 0] = len(x)
        for i in range(num_windows):
            in_sli = slice(i * self.params.stride_in, i * self.params.stride_in + self.params.kernel_size_in)
            out_sli = slice(i * self.params.stride_out, i * self.params.stride_out + self.params.kernel_size_out)
            out[out_sli, 0] = np.minimum(out[out_sli, 0], in_sli.start)
            out[out_sli, 1] = np.maximum(out[out_sli, 1], in_sli.stop)

        return out
