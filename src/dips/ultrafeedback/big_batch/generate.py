import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from dips.ultrafeedback.big_batch.config import Args
from dips.ultrafeedback.big_batch.tokenization_utils import swap_eos_token
from vllm import LLM, SamplingParams
from typing import List
from accelerate import Accelerator
import gc
import argparse
from peft import PeftModel
import pickle as pkl
import os

def forward(model: AutoModelForCausalLM, 
            input_ids: torch.Tensor, 
            tokenizer: AutoTokenizer, 
            args: Args,
            ref: bool = False):
    """
    Get model output for given query_responses. 
    If ref is True, the model's adapter module is disabled (e.g peft models are reverted to their base models).
    """
    attention_mask = input_ids != tokenizer.pad_token_id
    input_ids = torch.masked_fill(input_ids, ~attention_mask, 0)
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

def generate_vllm(policy: AutoModelForCausalLM,
                  prompts: List[str],
                  tokenizer: AutoTokenizer, # Tokenizer used for prompts & padding
                  generation_config: GenerationConfig,
                  context_length: int, # Length of tokenized prompt
                  output_length: int, # Desired total output length (prompt + response)
                  device: torch.device,
                  args: Args,
                  n_outputs_per_prompt: int = 1,
                  save_dir: str = None,
                  get_logprobs: bool = False
                 ) -> torch.Tensor:
    """
    Generates sequences using VLLM engine.
    Designed to be called from the main process (rank 0).

    Args:
        policy: Model to use for generation.
        prompts: List of prompt strings.
        tokenizer: Tokenizer associated with the prompts/model.
        generation_config: HF GenerationConfig to extract parameters from.
        context_length: The max length of the tokenized prompts.
        output_length: The target total sequence length (prompt + response) for VLLM's max_tokens.
        device: Target torch device for the output tensor.
        n_outputs_per_prompt: Number of sequences per prompt (k).

    Returns:
        Tensor of generated sequences (prompt + response). Shape: [num_prompts * n_outputs_per_prompt, output_length]
    """

    # TODO: This is a hack to get the policy model to work with VLLM.
    policy_lora_merged = policy.merge_and_unload()
    policy_lora_merged.save_pretrained(f"{args.output_dir}/temp_lora_merged")

    del policy_lora_merged
    gc.collect()
    torch.cuda.empty_cache()

    llm_engine = LLM(
        model=f"{args.output_dir}/temp_lora_merged",
        tokenizer=tokenizer.name_or_path,
        max_model_len=output_length,
        tensor_parallel_size=args.world_size,
        gpu_memory_utilization=0.65,
        trust_remote_code=True,
    )
    
    sampling_params = SamplingParams(
        n=n_outputs_per_prompt,
        temperature=generation_config.temperature if generation_config.temperature > 1e-6 else 1e-6, # VLLM requires temp > 0
        top_p=generation_config.top_p if generation_config.top_p < 1.0 else 1.0,
        top_k=generation_config.top_k if generation_config.top_k > 0 else -1, # VLLM uses -1 for no top_k
        max_tokens=generation_config.max_new_tokens,
        # min_tokens=generation_config.min_new_tokens, # VLLM might not support min_tokens
        stop_token_ids=[tokenizer.eos_token_id] if tokenizer.eos_token_id else None,
        skip_special_tokens=False, # Keep special tokens like EOS
        logprobs=None if not get_logprobs else 1, # Not needed here; will compute later with `forward`
    )

    # VLLM call
    vllm_outputs = llm_engine.generate(prompts, sampling_params, use_tqdm=True)

    all_output_sequences = []
    all_logprobs = []
    # Re-tokenize prompts to get the exact input IDs VLLM used (more robust than assuming first N tokens match)
    prompt_token_ids_dict = tokenizer(prompts, return_tensors="pt", padding="max_length", truncation=True, max_length=context_length)
    prompt_token_ids = prompt_token_ids_dict.input_ids

    for request_output in vllm_outputs:
        for completion in request_output.outputs:
            generated_token_ids = torch.tensor(completion.token_ids, device=device)
            if get_logprobs:
                generated_token_logprobs = torch.tensor(completion.logprobs, device=device)

            # If the generated token ids are shorter than the maximum length, pad to max length (match behavior of generate())
            if generated_token_ids.shape[0] < generation_config.max_new_tokens:
                padding = torch.full(
                    (generation_config.max_new_tokens - generated_token_ids.shape[0],),
                    tokenizer.pad_token_id,
                    dtype=torch.long,
                    device=device
                )
                generated_token_ids = torch.cat([generated_token_ids, padding])
                if get_logprobs:
                    logprob_padding = torch.full(
                        (generation_config.max_new_tokens - generated_token_logprobs.shape[0],),
                        0.0,
                        dtype=torch.float,
                        device=device
                    )
                    generated_token_logprobs = torch.cat([generated_token_logprobs, logprob_padding])

            all_output_sequences.append(generated_token_ids)
            if get_logprobs:
                all_logprobs.append(generated_token_logprobs)
    
    if not all_output_sequences:
        # Handle case where VLLM returns no output
        return torch.empty((0, output_length), dtype=torch.long, device=device)

    if get_logprobs:
        all_logprobs = torch.stack(all_logprobs) # [batch_size * n_outputs_per_prompt, output_length]
    final_tensor = torch.stack(all_output_sequences) # [batch_size * n_outputs_per_prompt, output_length]

    if args.swap_eos_token:
         # Apply token swapping if needed *after* generation
         final_tensor = swap_eos_token(final_tensor,
                                      from_token = "<|end_of_text|>",
                                      to_token = "<|eot_id|>")
         
    # ======= Memory Cleanup =======
    # Explicitly destroy the process group if it was initialized
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
        
    del llm_engine
    del vllm_outputs
    del all_output_sequences
    del prompt_token_ids
    gc.collect()
    torch.cuda.empty_cache()

    if save_dir:
        # Create the save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        torch.save(final_tensor, os.path.join(save_dir, "responses.pkl"))
    
    if get_logprobs:
        return final_tensor, all_logprobs
    else:
        return final_tensor

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--lora_dir", type=str, default="models/temp_lora")
    parser.add_argument("--output_length", type=int, default=1024)
    parser.add_argument("--context_length", type=int, default=1024)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--args_file", type=str, default=None)
    parser.add_argument("--prompts_file", type=str, default=None)
    parser.add_argument("--generation_config_file", type=str, default=None)
    parser.add_argument("--n_outputs_per_prompt", type=int, default=1)

    parsed_args = parser.parse_args()
    with open(parsed_args.args_file, "rb") as f:
        args = pkl.load(f)
    with open(parsed_args.prompts_file, "rb") as f:
        prompts = pkl.load(f)
    with open(parsed_args.generation_config_file, "rb") as f:
        generation_config = pkl.load(f)
    tokenizer = AutoTokenizer.from_pretrained(parsed_args.lora_dir)
    base_model = AutoModelForCausalLM.from_pretrained(parsed_args.base_model)
    policy = PeftModel.from_pretrained(base_model, parsed_args.lora_dir)
    print(f"Saving in {parsed_args.save_dir}")
    generate_vllm(policy = policy, 
                  prompts = prompts,
                  tokenizer = tokenizer, 
                  generation_config = generation_config,
                  context_length = int(parsed_args.context_length),
                  output_length = int(parsed_args.output_length),
                  device = parsed_args.device,
                  args = args, 
                  save_dir = parsed_args.save_dir,
                  n_outputs_per_prompt = int(parsed_args.n_outputs_per_prompt))