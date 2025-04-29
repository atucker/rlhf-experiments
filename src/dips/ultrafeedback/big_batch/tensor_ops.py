import torch
from accelerate.state import DistributedType
from transformers import AutoTokenizer
import numpy as np

# taken from https://github.com/microsoft/DeepSpeedExamples/blob/737c6740bec38b77a24a59135b6481a53d566b38/applications/DeepSpeed-Chat/training/utils/model/model_utils.py#L20C1-L26C52
def configure_dropout(model_config, dropout_layer_keys, dropout):
    if dropout is not None:
        for key in dropout_layer_keys:
            if hasattr(model_config, key):
                print(f"Setting model_config.{key} to {dropout}")
                setattr(model_config, key, dropout)
                
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.normal_(layer.weight, std=std)
    torch.nn.init.constant_(layer.bias, val=bias_const)
    return layer

def whiten(values, shift_mean=True, eps=1e-5):
    # `unbiased=False` matches TF `tf.nn.moments`'s setting
    # Normalize the values to have a mean of 0 (if shift_mean is false) and a variance of 1
    mean, var = torch.mean(values), torch.var(values, unbiased=False)
    whitened = (values - mean) * torch.rsqrt(var + eps)
    if not shift_mean:
        whitened += mean
    return whitened

def filter_by_length(sample: str, 
                     tokenizer: AutoTokenizer, 
                     max_length: int = 256,
                     ) -> torch.Tensor:
    return len(tokenizer(sample["instruction"]).input_ids) <= max_length

def debug_tensor_info(tensor, name, enabled = True):
    if enabled:
        print(f"{name}:")
        print(f"- Shape: {tensor.shape}")
        print(f"- Device: {tensor.device}")
        print(f"- Dtype: {tensor.dtype}")
        print(f"- Memory: {tensor.element_size() * tensor.nelement() / 1024 / 1024:.2f}MB")
        print(f"- Requires grad: {tensor.requires_grad}")

def first_true_indices(bools, dtype=torch.long) -> torch.Tensor:
    """
    Takes an N-dimensional bool tensor and returns an (N-1)-dimensional tensor of integers giving
    the position of the first True in each "row".

    Returns the length of the rows (bools.size(-1)) if no element is True in a given row.
    """
    row_len = bools.size(-1)
    zero_or_index = row_len * (~bools).type(dtype) + torch.arange(row_len, dtype=dtype, device=bools.device)
    return torch.min(zero_or_index, dim=-1).values


def truncate_response(args, tokenizer: AutoTokenizer, responses: torch.Tensor) -> torch.Tensor:
    """
    Truncates responses after the first occurrence of the truncate token.
    """
    trunc_idxs = first_true_indices(responses == args.task.truncate_token_id).unsqueeze(-1)
    new_size = [1] * (len(responses.size()) - 1) + [responses.shape[1]]
    idxs = torch.arange(responses.shape[1], device=responses.device).view(*new_size)
    postprocessed_responses = torch.masked_fill(responses, idxs > trunc_idxs, tokenizer.pad_token_id)
    return postprocessed_responses

def force_clear_grads(accelerator, model, optimizer):
    """
    Forces accelerate's accumulate() to clear its cached gradients. Called when keeping the gradients are unnecessary
    between sampling steps.
    """
    # Exit any no_sync states if they exist
    if accelerator.distributed_type == DistributedType.MULTI_GPU:
        if hasattr(model, '_no_sync_context'):
            model._no_sync_context.__exit__(None, None, None)
    
    optimizer.zero_grad()
    torch.cuda.empty_cache()