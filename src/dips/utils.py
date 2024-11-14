import random
import numpy as np
import torch

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def get_grad_norms(params: list, device = "cpu") -> torch.tensor:
    grad_norms = []
    for p in params:
        if p.grad is not None:
            grad_norms.append(p.grad.norm().item())
        else:
            grad_norms.append(0)
    grad_norms = torch.tensor(grad_norms, device = device)
    return grad_norms