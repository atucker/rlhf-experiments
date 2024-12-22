
import torch
import torch.amp
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    PretrainedConfig,
    PreTrainedModel,
)

# Source: dips/kl.py
class PrecisionWrapper(nn.Module):
    def __init__(self, model: PreTrainedModel):
        super().__init__()
        self.model = model
        self.lm_head = model.get_output_embeddings()
    
    def forward(self, **kwargs):
        before_unembed = self.model(**kwargs, output_hidden_states = True).hidden_states[-1]

        with torch.amp.autocast(device_type = "cuda", enabled = False):
            before_unembed = before_unembed.to(torch.float32)
            logits = self.lm_head(before_unembed)
        return logits
    

# Source: dips/kl.py
# taken from https://github.com/OpenLMLab/MOSS-RLHF/blob/40b91eb2f2b71b16919addede0341d2bef70825d/ppo/ppo_trainer.py#L29
# we did this we can do a single `model = accelerator.prepare(model)`
class PolicyAndValueWrapper(nn.Module):
    """
    A wrapper that fuses the model and its value function. Note that in this implementation the policy and value are both LORAs and share the same backbone.
    """
    def __init__(self, policy, critic) -> None:
        super().__init__()
        self.policy = policy
        self.critic = critic

    def forward(self, **kwargs):
        return self.policy(**kwargs), self.critic(**kwargs)

# Source: dips/kl.py
def exact_div(a, b):
    q = a // b
    if a != q * b:
        raise ValueError(f"Inexact division: {a} / {b} = {a / b}")
    return q