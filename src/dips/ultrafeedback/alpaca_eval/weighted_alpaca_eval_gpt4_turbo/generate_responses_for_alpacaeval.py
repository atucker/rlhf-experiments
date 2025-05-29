"""
Note: This script isn't useful - I wrote this before I realized that AlpacaEval had its own prompts. Whoops!
"""
from vllm import LLM, SamplingParams
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm, trange
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True)
args = parser.parse_args()

query_length = 256
response_length = 1024
template_length = 64
llm = LLM(model=args.model, 
          task="generate", 
          max_model_len = query_length + response_length + template_length + 1, 
          tensor_parallel_size = 2, 
          gpu_memory_utilization = 0.8)

ds = load_dataset("openbmb/UltraFeedback")

ultrafeedback_instructions = ds["train"]["instruction"]
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
tokenizer.add_special_tokens({"pad_token": "[PAD]"})
tokenizer.padding_side = "left"
pairs = [(tokenizer.apply_chat_template([{"role": "user", "content": instruction}],
                                                 add_generation_prompt=True,
                                                  padding = False,
                                                  max_length = query_length + template_length,
                                                  truncation = True), instruction) for instruction in tqdm(ultrafeedback_instructions)
                                                  if len(tokenizer(instruction).input_ids) <= query_length]

prompt_token_ids, instructions = zip(*pairs)
prompt_token_ids = list(prompt_token_ids)
print(f"Sampling {len(prompt_token_ids)} instructions.")

sampling_params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens = response_length)

CHECKPOINT_FREQ = 5000
alpaca_eval_outputs = []

for i in trange(0, len(prompt_token_ids), CHECKPOINT_FREQ):
    outputs = llm.generate(prompt_token_ids=prompt_token_ids[i:min(i+CHECKPOINT_FREQ, len(prompt_token_ids))], 
                           sampling_params=sampling_params)

    for index in range(len(outputs)):
        output = {"instruction": instructions[index], "output": outputs[index].outputs[0].text}
        alpaca_eval_outputs.append(output)

    with open(f"{args.model}_alpaca_eval_outputs_{i}.json", "w") as f:
        json.dump(alpaca_eval_outputs, f)

with open(f"{args.model}_alpaca_eval_outputs_final.json", "w") as f:
    json.dump(alpaca_eval_outputs, f)