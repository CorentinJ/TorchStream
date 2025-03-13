from typing import Union

import numpy as np
import torch


def get_nan_range(x: Union[torch.Tensor, np.ndarray], dim: int = -1):
    # TODO! doc
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()

    dim = range(x.ndim)[dim]
    x = x.mean(axis=tuple(i for i in range(x.ndim) if i != dim))

    corrupted_idx = np.where(np.isnan(x))[0]

    if not len(corrupted_idx):
        return None, None
    return corrupted_idx[0], corrupted_idx[-1] + 1
