from typing import List, Tuple

from torchstream.sliding_window.sliding_window_params import SlidingWindowParams


def _sli_params_to_test_args(params_iter) -> Tuple[List[Tuple[SlidingWindowParams, int, int]], List[str]]:
    out_args = []
    for params_dict in params_iter:
        try:
            # Skip these cases that are duplicates (dilation has no effect if the kernel size is 1)
            if (
                "dilation" in params_dict
                and params_dict["dilation"] > 1
                and params_dict.get("kernel_size_in", 1) == 1
                and params_dict.get("kernel_size_out", 1) == 1
            ):
                continue

            # Kernel size -> kernel span conversion for convolutions with dilation
            dilation = params_dict.pop("dilation", 1)
            params_dict["kernel_size_in"] = (params_dict.get("kernel_size_in", 1) - 1) * dilation + 1
            params_dict["kernel_size_out"] = (params_dict.get("kernel_size_out", 1) - 1) * dilation + 1

            sli_params = SlidingWindowParams(**params_dict)
            # The sli params only specify the span, we'll need to keep the dilation info for building conv layers
            out_args.append((sli_params, dilation))

        except ValueError:
            pass

    ids = [
        " ".join(
            filter(
                None,
                (
                    f"ki={p.kernel_size_in}(d{dilation})" if p.kernel_size_in > 1 else None,
                    f"si={p.stride_in}" if p.stride_in > 1 else None,
                    f"p=({p.left_pad}, {p.right_pad})" if p.left_pad + p.right_pad > 0 else None,
                    f"ko={p.kernel_size_out}(d{dilation})" if p.kernel_size_out > 1 else None,
                    f"so={p.stride_out}" if p.stride_out > 1 else None,
                    f"trim={p.out_trim}" if p.out_trim > 0 else None,
                ),
            )
        )
        for p, dilation in out_args
    ]

    return out_args, ids


CONV_1D_PARAMS = _sli_params_to_test_args(
    (
        dict(
            kernel_size_in=kernel_size,
            stride_in=stride,
            left_pad=padding[0],
            right_pad=padding[1],
            dilation=dilation,
        )
        for kernel_size in [1, 2, 3, 10, 17]
        for stride in [1, 2, 3, 10, 17]
        for padding in [(0, 0), (1, 1), (2, 0), (0, 3), (1, 4)]
        for dilation in [1, 2, 3]
    )
)


TRANSPOSED_CONV_1D_PARAMS = _sli_params_to_test_args(
    (
        dict(
            kernel_size_out=kernel_size,
            stride_out=stride,
            out_trim=out_trim,
            dilation=dilation,
        )
        for kernel_size in [1, 2, 3, 10, 17]
        for stride in [1, 2, 3, 10, 17]
        for out_trim in [0, 1, 2, 8, 9]
        for dilation in [1, 2, 3]
    )
)

# TODO: reduce these:
#  - Avoid cases equiv to conv or tconv
#  - Keep only unique values
MOVING_AVERAGE_PARAMS = _sli_params_to_test_args(
    (
        dict(
            kernel_size_in=kernel_size_in,
            stride_in=stride_in,
            left_pad=padding[0],
            right_pad=padding[1],
            kernel_size_out=kernel_size_out,
            stride_out=stride_out,
            out_trim=out_trim,
        )
        for kernel_size_in in [1, 2, 5, 10]
        for stride_in in [1, 2, 3]
        for padding in [(0, 0), (3, 0), (1, 2)]
        for kernel_size_out in [1, 2, 4, 7]
        for stride_out in [1, 2, 7]
        for out_trim in [0, 1, 3, 6]
    )
)

EDGE_CASES_PARAMS = _sli_params_to_test_args(
    [
        dict(),
        dict(kernel_size_out=5, stride_out=5, out_trim=4),
        dict(kernel_size_in=2, stride_in=2, left_pad=1, right_pad=1, kernel_size_out=2, stride_out=2, out_trim=1),
        dict(kernel_size_out=7, out_trim=5),
        dict(kernel_size_out=10, stride_out=3, out_trim=8),
        dict(kernel_size_in=5, stride_in=1, left_pad=3, right_pad=0, kernel_size_out=7, stride_out=2, out_trim=6),
        dict(kernel_size_in=3, stride_in=1, left_pad=0, right_pad=2, kernel_size_out=2, stride_out=2, out_trim=1),
        dict(kernel_size_in=31, stride_in=17),
        # TODO Add dilation=2 (k=17)
        dict(kernel_size_in=33, stride_in=17, left_pad=2),
        # TODO Add dilation=2 (k=10)
        dict(kernel_size_in=19, stride_in=10, left_pad=1, right_pad=4),
    ]
)
