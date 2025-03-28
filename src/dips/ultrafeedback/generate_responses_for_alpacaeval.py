from vllm import LLM, SamplingParams
llm = LLM(model="merged_model", task="generate", max_model_len = 1024 + 64 + 1, tensor_parallel_size = 4)

from datasets import load_dataset
ds = load_dataset("openbmb/UltraFeedback")

from transformers import AutoTokenizer

ultrafeedback_instructions = ds["train"]["instruction"]
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
tokenizer.add_special_tokens({"pad_token": "[PAD]"})
prompt_token_ids = [tokenizer.apply_chat_template([{"role": "user", "content": instruction}],
                                                 add_generation_prompt=True,
                                                  padding = "max_length",
                                                  max_length = 1024 + 64,
                                                  truncation = True) for instruction in ultrafeedback_instructions]

sampling_params = SamplingParams(temperature=0.7, top_p=0.95)

outputs = llm.generate(prompt_token_ids=prompt_token_ids, sampling_params=sampling_params)