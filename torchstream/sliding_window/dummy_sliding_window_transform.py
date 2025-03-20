import numpy as np

from torchstream.sliding_window.sliding_window_params import SlidingWindowParams


class DummySlidingWindowTransform:
    def __init__(self, params: SlidingWindowParams):
        self.params = params

    def __call__(self, x: np.ndarray):
        left_pad, right_pad = self.params.get_padding(x.shape[-1])
        out_size = self.params.get_output_size(x.shape[-1])
        num_wins = self.params.get_num_windows(x.shape[-1])

        x = np.concatenate([np.zeros(left_pad), x])
        if right_pad < 0:
            x = x[:right_pad]
        else:
            x = np.concatenate([x, np.zeros(right_pad)])

        out = np.zeros((out_size, 2), dtype=np.int64)
        out[:, 0] = len(x)
        for i in range(num_wins):
            in_sli = slice(i * self.params.stride_in, i * self.params.stride_in + self.params.kernel_size_in)
            out_sli = slice(i * self.params.stride_out, i * self.params.stride_out + self.params.kernel_size_out)
            out[out_sli, 0] = np.minimum(out[out_sli, 0], in_sli.start)
            out[out_sli, 1] = np.maximum(out[out_sli, 1], in_sli.stop)

        return out
