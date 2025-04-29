import random
import numpy as np
import torch

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def get_grad_norms(loss, params: list, device = "cpu") -> torch.Tensor:
    grad = torch.autograd.grad(
        outputs = loss,
        inputs = params,
        create_graph = False,
        retain_graph = True,
    )
    grad_norms = torch.tensor([g.norm() for g in grad])
    return grad_norms