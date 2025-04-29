import torch
import torch.nn as nn
from transformers import AutoTokenizer

def get_reward(reward_model: nn.Module, 
               input_ids: torch.Tensor,
               tokenizer: AutoTokenizer,
               ):
    """
    Uses the reward model to calculate reward information for the given query_responses.

    Returns a scalar reward for each query_response pair.
    Expected input shape: [batch_size, seq_len] (should include both prompt and response, inc. chat template)
    """
    with torch.no_grad():
        scores = []
        reward_breakdown = []
        reward_breakdown_coeffs = []
        for elem in input_ids:
            no_pad_input_ids = torch.masked_select(elem, elem != tokenizer.pad_token_id).unsqueeze(0)
            attention_mask = no_pad_input_ids != tokenizer.pad_token_id
            output = reward_model(input_ids=no_pad_input_ids, 
                                attention_mask=attention_mask,
                                return_dict=True)
            scores.append(output.score)
            reward_breakdown.append(output.rewards)
            reward_breakdown_coeffs.append(output.gating_output @ reward_model.reward_transform_matrix.data.T)
        return torch.cat(scores), torch.cat(reward_breakdown), torch.cat(reward_breakdown_coeffs)