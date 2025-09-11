import numpy as np

from torchstream.sliding_window.sliding_window_params import SlidingWindowParams


class DummySlidingWindowTransform:
    """
    This class is a trivial implementation of a sliding window transform. It's a moving average with a sum reduction
    for overlapping output windows.
    The class is used for empirical verification of sliding window related implementations.
    """

    def __init__(self, params: SlidingWindowParams):
        self.params = params

    def __call__(self, x: np.ndarray):
        (left_pad, right_pad), num_wins, out_size = self.params.get_metrics_for_input(x.shape[-1])

        out = np.zeros(out_size)
        for (in_start, in_stop), (out_start, out_stop) in self.params.iter_bounded_kernel_map(x.shape[-1]):
            # NOTE: not weighing in the padding, but this is just a dummy transform so it's fine
            out[out_start:out_stop] += np.mean(x[in_start:in_stop])

        return out
