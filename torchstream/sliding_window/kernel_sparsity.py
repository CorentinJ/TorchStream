import numpy as np


def get_init_kernel_array(kernel_size: int) -> np.ndarray:
    """
    Initialize a default kernel sparsity array.

    A kernel array is an integer numpy array with the following convention:
    - 0: the kernel does not cover the element at that index
    - 1: unknown (may or may not cover the element)
    - 2: the kernel covers the element at that index

    By default, we set all elements to 1 (unknown) and force the first and last
    elements to 2 (covered) to ensure a minimum span.

    :param kernel_size: Span/size of the kernel.
    :return: A kernel sparsity array initialized with 1s, with the first and last elements set to 2.
    """
    kernel = np.ones(int(kernel_size), dtype=np.int64)
    if kernel.size > 0:
        kernel[0] = kernel[-1] = 2
    return kernel
