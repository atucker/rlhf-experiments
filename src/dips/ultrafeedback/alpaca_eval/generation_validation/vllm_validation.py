import argparse
import torch
from transformers import AutoTokenizer
from datasets import load_dataset
from vllm import LLM, SamplingParams

def main(args):
    query_length = 256
    response_length = 1024
    template_length = 64
    max_model_len = query_length + response_length + template_length

    # --- Load dataset and instruction ---
    print("Loading dataset...")
    ds = load_dataset("openbmb/UltraFeedback")
    # Using the same instruction index as in the notebook example
    sample_instruction = ds["train"]["instruction"][args.instruction_index]
    print(f"Using instruction index: {args.instruction_index}")

    # --- Load Tokenizer ---
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    # Add pad token if necessary (check if the base model already has it)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        # Resize model embeddings if pad token was added, VLLM might handle this automatically
        # when loading, but good practice to be aware.
    tokenizer.padding_side = "left"

    # --- Prepare Prompt ---
    print("Preparing prompt...")
    prompt_token_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": sample_instruction}],
        add_generation_prompt=True,
        padding="max_length", # Pad to ensure consistent input length for the model if needed
        max_length=query_length + template_length, # Match notebook's max_length for the prompt part
        truncation=True,
        return_tensors="pt"
    )[0].tolist() # VLLM needs a list of token IDs

    # --- Initialize VLLM ---
    print(f"Loading model {args.merged_model_dir} with VLLM...")
    # Adjust tensor_parallel_size and gpu_memory_utilization as needed
    llm = LLM(
        model=args.merged_model_dir,
        tokenizer=args.tokenizer_path, # Use the same tokenizer path
        max_model_len=max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True # Often needed for custom models/tokenizers
    )

    # --- Set Sampling Parameters ---
    # Match the response length from the notebook
    sampling_params = SamplingParams(
        temperature=0.75, # Set to 0 for deterministic output like the original script might implicitly do
        top_p=1.0,       # Disable top-p sampling
        max_tokens=response_length
    )

    # --- Generate Response ---
    print("Generating response...")
    outputs = llm.generate(prompt_token_ids=[prompt_token_ids], sampling_params=sampling_params)

    # --- Print Results ---
    print("\nInstruction: \n=======================================\n", sample_instruction)
    # Outputs is a list of RequestOutput objects
    generated_text = outputs[0].outputs[0].text
    print("\nResponse: \n=======================================\n", generated_text)
    print("\nGeneration finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate a merged model using VLLM.")
    parser.add_argument(
        "--merged_model_dir",
        type=str,
        required=True,
        help="Directory containing the merged model weights and config.",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Path to the tokenizer.",
    )
    parser.add_argument(
        "--instruction_index",
        type=int,
        default=1,
        help="Index of the instruction to use from the UltraFeedback train split.",
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help="Number of GPUs to use for tensor parallelism.",
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.8,
        help="GPU memory utilization fraction.",
    )
    args = parser.parse_args()
    main(args) 