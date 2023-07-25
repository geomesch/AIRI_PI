import torch
import numpy as np
import random

def set_seed(seed):
    torch.use_deterministic_algorithms(True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def calc_mask(K, dilation=0, eps=1e-2):
    mask = K != 0
    if dilation:
        with torch.no_grad():
            if dilation == 'precond':
                ind = tuple(1 for _ in range(len(K.shape) - 1))
                mask = K / K.flatten(1).max(1)[0].reshape(-1, *ind) + eps
            else:
                mask = torch.nn.functional.max_pool2d(K, dilation, padding=dilation // 2, stride=1).bool()
    return mask