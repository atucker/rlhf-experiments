import torch
from transformers import AutoTokenizer
from typing import List
from dips.ultrafeedback.big_batch.config import Args

def swap_eos_token(sequences: torch.Tensor, 
                   tokenizer: AutoTokenizer,
                   from_token: str = "<|eot_id|>", 
                   to_token: str = "<|end_of_text|>") -> torch.Tensor:
    from_token_id = tokenizer.convert_tokens_to_ids(from_token)
    to_token_id = tokenizer.convert_tokens_to_ids(to_token)
    return torch.where(sequences == from_token_id, to_token_id, sequences)

def maybe_use_chat_template(instruction: List[str], 
                            use_chat_template: bool, 
                            tokenizer: AutoTokenizer,
                            args: Args,
                            device: torch.device) -> torch.Tensor:
    if use_chat_template:
        messages = [[{"role": "user", "content": instruction}] for instruction in instruction]
        # ^ necessary to use apply_chat_template
        queries = tokenizer.apply_chat_template(messages, 
                                                padding = "max_length",
                                                return_tensors="pt",
                                                add_generation_prompt = True,
                                                max_length = args.task.query_length + args.task.chat_template_buffer_length,
                                                truncation = True,
        )
        return queries.to(device)
    else:
        return tokenizer(instruction, 
                         padding="max_length", 
                         max_length=args.task.query_length,
                         truncation=True,
                         return_tensors="pt",
        ).input_ids.to(device)