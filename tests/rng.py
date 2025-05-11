import random
from typing import Optional

import numpy as np
import torch
import z3


def set_seed(seed: Optional[int]):
    if not seed:
        return

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    random.seed(seed)

    np.random.seed(seed)

    # FIXME: doesn't work in getting deterministic testing?
    z3.set_param("smt.random_seed", seed)
    z3.set_param("smt.arith.random_initial_value", False)
