import random

import numpy as np
import torch


def set_random_seed(seed: int | None):
    if seed is None:
        return

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if hasattr(torch, "cuda") and hasattr(torch.cuda, "manual_seed_all"):
        torch.cuda.manual_seed_all(seed)
