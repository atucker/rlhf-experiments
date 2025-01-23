import argparse
from typing import List, Dict, Any
import json
from vllm import LLM, SamplingParams
from openai import OpenAI
from tqdm import tqdm
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import logging
from datasets import load_dataset
import torch
import os
from torch import multiprocessing

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
multiprocessing.set_start_method('spawn', force=True)

client = OpenAI(
    api_key = os.environ.get("OPENAI_API_KEY"),
)
def setup_torch_compile():
    """Configure torch compilation settings."""
    torch._dynamo.config.suppress_errors = True
    torch._inductor.config.fallback_random = True
    # Disable TorchInductor if CUDA setup is incomplete
    if not torch.cuda.is_available():
        torch._dynamo.config.backend = "eager"
    os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0"  # Adjust based on your GPU

def load_ultrafeedback_prompts(num_samples: int = 100) -> List[str]:
    """Load evaluation prompts from UltraFeedback dataset."""
    dataset = load_dataset("openbmb/UltraFeedback", split="train")
    # Sample random subset if num_samples specified
    if num_samples and num_samples < len(dataset):
        dataset = dataset.shuffle(seed=42).select(range(num_samples))
    return [item["instruction"] for item in dataset]

def setup_vllm_model(model_path: str, tensor_parallel_size: int = 1) -> LLM:
    """Initialize VLLM model with specified parameters."""
    setup_torch_compile()
    return LLM(
        model=model_path,
        tensor_parallel_size=tensor_parallel_size,
        trust_remote_code=True,
        gpu_memory_utilization=0.45,
    )

def load_prompts(prompt_file: str) -> List[str]:
    """Load evaluation prompts from file."""
    with open(prompt_file, 'r') as f:
        return json.load(f)

def get_model_responses(
    model: LLM, 
    prompts: List[str], 
    sampling_params: SamplingParams
) -> List[str]:
    """Generate responses using VLLM for a batch of prompts."""
    outputs = model.generate(prompts, sampling_params)
    return [output.outputs[0].text for output in outputs]

def get_comparison_messages(prompt: str, response_a: str, response_b: str) -> List[dict]:
    system_prompt = "You are a helpful, harmless, and precise assistant for checking the quality of a pair of given responses."

    user_string = f"""Given the following prompt and two responses, which response is better?

    Prompt: {prompt}

    Response A: {response_a}

    Response B: {response_b}

    Please provide your judgment as a single character:
    'A' if Response A is better
    'B' if Response B is better
    'T' if they are of equal quality

    Your response:"""

    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_string}]

def get_eval_judgement(prompt: str, response_a: str, response_b: str, model: str = "gpt-4o") -> str:
    """Get OpenAI model judgment on which response is better."""
    comparison_messages = get_comparison_messages(prompt, response_a, response_b)
    try:
        response = client.chat.completions.create(
            model=model,
            messages=comparison_messages,
            temperature=0.0,
            max_tokens=1,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Error getting GPT-4 judgment: {e}")
        return "ERROR"

def calculate_winrate(judgments: List[str]) -> Dict[str, float]:
    """Calculate winrate statistics from judgments."""
    total = len(judgments)
    wins_a = judgments.count('A')
    wins_b = judgments.count('B')
    ties = judgments.count('T')
    errors = judgments.count('ERROR')
    
    valid_comparisons = total - errors
    
    return {
        'model_a_winrate': wins_a / valid_comparisons if valid_comparisons > 0 else 0,
        'model_b_winrate': wins_b / valid_comparisons if valid_comparisons > 0 else 0,
        'tie_rate': ties / valid_comparisons if valid_comparisons > 0 else 0,
        'error_rate': errors / total if total > 0 else 0,
        'total_comparisons': total
    }

def main():
    parser = argparse.ArgumentParser(description='Run winrate evaluation using VLLM and GPT-4')
    parser.add_argument('--model-a-path', required=True, help='Path to first model')
    parser.add_argument('--model-b-path', required=True, help='Path to second model')
    parser.add_argument('--output-file', required=True, help='Path to save results')
    parser.add_argument('--tensor-parallel-size', type=int, default=1, help='Number of GPUs to use for tensor parallelism')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for VLLM inference')
    parser.add_argument('--num-samples', type=int, default=100, help='Number of prompts to sample from UltraFeedback')

    args = parser.parse_args()

    # Load prompts
    prompts = load_ultrafeedback_prompts(args.num_samples)
    logger.info(f"Loaded {len(prompts)} prompts for evaluation")

    # Setup VLLM models
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.95,
        max_tokens=512
    )

    logger.info("Initializing models...")
    model_a = setup_vllm_model(args.model_a_path, args.tensor_parallel_size)
    model_b = setup_vllm_model(args.model_b_path, args.tensor_parallel_size)

    # Generate responses
    logger.info("Generating responses from Model A...")
    responses_a = get_model_responses(model_a, prompts, sampling_params)
    logger.info("Generating responses from Model B...")
    responses_b = get_model_responses(model_b, prompts, sampling_params)

    # Get GPT-4 judgments
    logger.info("Getting GPT-4 judgments...")
    judgments = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        future_to_prompt = {
            executor.submit(get_eval_judgement, prompt, resp_a, resp_b): i
            for i, (prompt, resp_a, resp_b) in enumerate(zip(prompts, responses_a, responses_b))
        }
        
        for future in tqdm(future_to_prompt, total=len(prompts)):
            judgment = future.result()
            judgments.append(judgment)

    # Calculate and save results
    results = calculate_winrate(judgments)
    
    # Create DataFrame from detailed results
    df = pd.DataFrame({
        'prompt': prompts,
        'response_a': responses_a, 
        'response_b': responses_b,
        'judgment': judgments
    })
    
    # Save both detailed DataFrame and aggregate metrics
    output = {
        'config': vars(args),
        'aggregate_metrics': results,
    }
    
    # Save JSON with config and metrics
    with open(args.output_file, 'w') as f:
        json.dump(output, f, indent=2)
        
    # Save DataFrame to CSV
    df_path = args.output_file.replace('.json', '_details.csv')
    df.to_csv(df_path, index=False)
    
    logger.info(f"Results saved to {args.output_file}")
    logger.info(f"Detailed results saved to {df_path}")
    logger.info(f"Winrate results: {results}")

if __name__ == "__main__":
    main()