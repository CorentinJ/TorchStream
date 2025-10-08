from torchstream.sliding_window.sliding_window_params import SlidingWindowParams

SLI_EDGE_CASES = [
    SlidingWindowParams(),
    SlidingWindowParams(kernel_size_out=5, stride_out=5, out_trim=4),
    SlidingWindowParams(
        kernel_size_in=2, stride_in=2, left_pad=1, right_pad=1, kernel_size_out=2, stride_out=2, out_trim=1
    ),
    SlidingWindowParams(kernel_size_out=7, out_trim=5),
    # TODO? Add dilation=2 kernel
    SlidingWindowParams(kernel_size_in=33, stride_in=17, left_pad=2),
]
