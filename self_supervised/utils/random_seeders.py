import random

import torch
import numpy as np


def set_random_seeds(random_seed=0):
    r"""Sets the seed for generating random numbers.

    Args:
        random_seed: Desired random seed.
    """
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)
