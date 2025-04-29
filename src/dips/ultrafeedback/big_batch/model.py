import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from dips.ultrafeedback.big_batch.tokenization_utils import swap_eos_token
from dips.ultrafeedback.big_batch.config import Args, SamplingParams
from vllm import LLM
from typing import List

class PrecisionModel(AutoModelForCausalLM):
    def forward(self, *args, **kwargs):
        before_unembed = super().forward(*args, **kwargs, output_hidden_states = True).hidden_states[-1]
        with torch.amp.autocast(device_type = "cuda", enabled = False):
            before_unembed = before_unembed.to(torch.float32)
            logits = self.lm_head(before_unembed)
        return logits
    
def generate(lm_backbone: AutoModelForCausalLM, 
             queries: torch.Tensor, 
             tokenizer: AutoTokenizer, 
             generation_config: GenerationConfig,
             args: Args,
             n_outputs_per_prompt: int = 1) -> torch.Tensor:
    """
    Generates in a way that does not affect padding tokens.

    Args:
        lm_backbone: The language model backbone to use.
        queries: The queries to generate responses for. Shape: [batch_size, seq_len]
        tokenizer: The tokenizer to use
        generation_config: The generation configuration to use
        n_outputs_per_prompt: The number of outputs to generate per prompt (k in RLOO)

    Returns: 
        Generated responses. Shape: [batch_size * n_outputs_per_prompt, seq_len]
    """
    context_length = queries.shape[1]
    attention_mask = queries != tokenizer.pad_token_id
    input_ids = torch.masked_fill(queries, ~attention_mask, 0)
    output = lm_backbone.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        # position_ids=attention_mask.cumsum(1) - attention_mask.long(), # generation collapsed if this was turned on. TODO: why does generation collapse with this?
        generation_config=generation_config,
        return_dict_in_generate=True,
        num_return_sequences=n_outputs_per_prompt,
        return_legacy_cache = True,
        # output_scores = True,
    )
    expanded_queries = queries.repeat_interleave(n_outputs_per_prompt, dim=0) # [batch_size * n_outputs_per_prompt, seq_len]
    if args.swap_eos_token:
        output.sequences = swap_eos_token(output.sequences,
                                          from_token = "<|end_of_text|>",
                                          to_token = "<|eot_id|>")
    full_sequences = torch.cat((expanded_queries, output.sequences[:, context_length:]), dim=1)
    return full_sequences

def generate_vllm(llm_engine: LLM,
                  prompts: List[str],
                  tokenizer: AutoTokenizer, # Tokenizer used for prompts & padding
                  generation_config: GenerationConfig,
                  context_length: int, # Length of tokenized prompt
                  output_length: int, # Desired total output length (prompt + response)
                  device: torch.device,
                  args: Args,
                  n_outputs_per_prompt: int = 1,
                 ) -> torch.Tensor:
    """
    Generates sequences using VLLM engine.
    Designed to be called from the main process (rank 0).

    Args:
        llm_engine: Initialized VLLM LLM engine.
        prompts: List of prompt strings.
        tokenizer: Tokenizer associated with the prompts/model.
        generation_config: HF GenerationConfig to extract parameters from.
        context_length: The max length of the tokenized prompts.
        output_length: The target total sequence length (prompt + response) for padding.
        device: Target torch device for the output tensor.
        n_outputs_per_prompt: Number of sequences per prompt (k).

    Returns:
        Tensor of generated sequences (prompt + response). Shape: [num_prompts * n_outputs_per_prompt, output_length]
    """
    sampling_params = SamplingParams(
        n=n_outputs_per_prompt,
        temperature=generation_config.temperature if generation_config.temperature > 1e-6 else 1e-6, # VLLM requires temp > 0
        top_p=generation_config.top_p if generation_config.top_p < 1.0 else 1.0,
        top_k=generation_config.top_k if generation_config.top_k > 0 else -1, # VLLM uses -1 for no top_k
        max_tokens=generation_config.max_new_tokens,
        # min_tokens=generation_config.min_new_tokens, # VLLM might not support min_tokens
        stop_token_ids=[tokenizer.eos_token_id] if tokenizer.eos_token_id else None,
        skip_special_tokens=False, # Keep special tokens like EOS
        logprobs=None, # Not needed here; will compute later with `forward`
    )

    # VLLM call
    vllm_outputs = llm_engine.generate(prompts, sampling_params, use_tqdm=False)

    all_output_sequences = []
    # Re-tokenize prompts to get the exact input IDs VLLM used (more robust than assuming first N tokens match)
    prompt_token_ids_dict = tokenizer(prompts, return_tensors="pt", padding="max_length", truncation=True, max_length=context_length)
    prompt_token_ids = prompt_token_ids_dict.input_ids
    prompt_attn_mask = prompt_token_ids_dict.attention_mask

    output_idx = 0
    for i, request_output in enumerate(vllm_outputs):
        # Get the actual prompt tokens used (handling padding)
        current_prompt_len = prompt_attn_mask[i].sum().item()
        unpadded_prompt_tokens = prompt_token_ids[i, :current_prompt_len].to(device)

        for completion in request_output.outputs:
            generated_token_ids = torch.tensor(completion.token_ids, device=device)
            full_sequence = torch.cat([unpadded_prompt_tokens, generated_token_ids], dim=0)

            # Pad sequence to the maximum expected length (context + max_new_tokens)
            pad_len = output_length - full_sequence.shape[0]
            if pad_len < 0:
                # Generated sequence is longer than required output length, truncate
                full_sequence = full_sequence[:output_length]
                pad_len = 0
            elif pad_len > 0:
                 # Pad if shorter
                padding = torch.full((pad_len,), tokenizer.pad_token_id, dtype=full_sequence.dtype, device=device)
                full_sequence = torch.cat([full_sequence, padding], dim=0)

            all_output_sequences.append(full_sequence)
            output_idx += 1

    if not all_output_sequences:
        # Handle case where VLLM returns no output
        return torch.empty((0, output_length), dtype=torch.long, device=device)

    final_tensor = torch.stack(all_output_sequences) # [batch_size * n_outputs_per_prompt, output_length]

    if args.swap_eos_token:
         # Apply token swapping if needed *after* generation
         final_tensor = swap_eos_token(final_tensor,
                                      from_token = "<|end_of_text|>",
                                      to_token = "<|eot_id|>")
    return final_tensor

def forward(model: AutoModelForCausalLM, 
            responses: torch.Tensor, 
            tokenizer: AutoTokenizer, 
            args: Args,
            ref: bool = False):
    """
    Get model output for given query_responses. 
    If ref is True, the model's adapter module is disabled (e.g peft models are reverted to their base models).
    """
    attention_mask = responses != tokenizer.pad_token_id
    input_ids = torch.masked_fill(responses, ~attention_mask, 0)
    if args.swap_eos_token:
        input_ids = swap_eos_token(input_ids,
                                    from_token = "<|eot_id|>",
                                    to_token = "<|end_of_text|>")
    if ref:
        with model.disable_adapter():
            return model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
                output_hidden_states=True,
            )
    else:
        return model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            output_hidden_states=True,
        )