import random
import numpy as np
import torch

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

# def get_grad_norms(params: list, device = "cpu") -> torch.tensor:
#     grad_norms = []
#     for p in params:
#         if p.grad is not None:
#             grad_norms.append(p.grad.norm().item())
#         else:
#             grad_norms.append(0)
#     grad_norms = torch.tensor(grad_norms, device = device)
#     return grad_norms

def get_grad_norms(loss, params: list, device = "cpu") -> torch.tensor:
    grad = torch.autograd.grad(
        outputs = loss,
        inputs = params,
        create_graph = False,
        retain_graph = True,
    )
    grad_norms = torch.tensor([g.norm() for g in grad])
    return grad_norms