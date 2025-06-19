# TODO: Port updates to kl.py over

import os
import time

import numpy as np
import torch
import tyro
import wandb
from tqdm import trange

# Logging
from rich.console import Console
from rich.pretty import pprint

# Model
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
)
import warnings
import pickle as pkl

# Package imports
from dips.tldr.utils import set_seed
from dips.ultrafeedback.big_batch.config import Args
from dips.ultrafeedback.big_batch.tokenization_utils import maybe_use_chat_template
from dips.ultrafeedback.big_batch.eval import evaluate
from dips.ultrafeedback.big_batch.misc import AdaptiveKLController
from dips.ultrafeedback.big_batch.logging_utils import GradNormLogger, print_rich_table, parse_reward_breakdown_attributes
from dips.ultrafeedback.big_batch.tensor_ops import configure_dropout, truncate_response, first_true_indices, debug_tensor_info, force_clear_grads
from dips.ultrafeedback.big_batch.model import initialize_policy_with_optimizer
from dips.ultrafeedback.big_batch.data import get_dataloaders
from dips.ultrafeedback.big_batch.generate import forward
from dips.ultrafeedback.big_batch.reward import get_reward
import subprocess

wandb.login(key=os.environ["WANDB_API_KEY"])

if __name__ == "__main__":

    # ========= Setup =========
    args = tyro.cli(Args)

    local_seed = args.seed + 100003  # Prime
    set_seed(local_seed)
    torch.backends.cudnn.deterministic = True

    args.world_size = 1
    args.batch_size = args.per_device_train_batch_size * args.world_size

    if ("instruct" in args.base_model.lower()) and (not args.use_chat_template):
        warnings.warn("You are using an instruct model without chat template. This may lead to unexpected results.")
    
    if ("instruct" not in args.base_model.lower()):
        assert args.swap_eos_token, "Make sure to swap the eos token when using a non-instruct model!"
        if args.use_chat_template:
            warnings.warn("You are using a non-instruct model with chat template. This may lead to unexpected results; the tokenization scheme between the instruct model and the base model may be different.")

    if args.kl_grad_patch:
        assert not args.train_dips, "KL grad patch is only supported for RLOO training"

    train_is_multi_batch = args.per_device_train_batch_size != 1
    eval_is_multi_batch = (args.rloo_k * args.local_rollout_forward_batch_size) != 1
    
    if args.train_dips:
        if train_is_multi_batch != eval_is_multi_batch:
            warnings.warn("""One of your training or evaluation batch sizes is 1 while the other is not.
        It's a known issue that huggingface models generate slightly different logits depending on batch size. While this difference is slight,
        it completely breaks the probability weighting ratio for DIPS. For Llama-8b, all batch sizes != 1 have identical behavior.
        """)

    args.ppo.num_updates = args.total_episodes // args.batch_size

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        padding_side="left",
        trust_remote_code=True,
    )
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    if args.chat_template_tokenizer is not None:
        chat_template_tokenizer = AutoTokenizer.from_pretrained(
            args.chat_template_tokenizer,
            use_fast = True,
            trust_remote_code = True,
            padding_side="left"
        )
        chat_template_tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    else:
        chat_template_tokenizer = tokenizer
        
    # we use the padding token manually but do not resize the token embedding of the model
    if args.task.truncate_token == "eos":
        args.task.truncate_token_id = tokenizer.eos_token_id

    # ========= Logging =========
    console = Console(force_terminal=True)
    grad_norm_logger = GradNormLogger()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ========= Model =========
    model_config = AutoConfig.from_pretrained(args.base_model)
    configure_dropout(model_config, args.dropout_layer_keys, 0.0)  # disable dropout
    assert args.reward_model_path, "reward_model_path must be provided"
    reward_model: PreTrainedModel = AutoModelForSequenceClassification.from_pretrained(
        args.reward_model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map = device,
        # low_cpu_mem_usage=True,
    )
    pprint(model_config)
    pprint(reward_model.config)

    policy, optimizer, scheduler = initialize_policy_with_optimizer(args = args, 
                                                                    model_config = model_config, 
                                                                    grad_norm_logger = grad_norm_logger)
    del optimizer, scheduler

    # ========= Data =========
    dataloader, validation_dataloader = get_dataloaders(args, tokenizer)

    # sync random states for DataLoader(shuffle=True) before `accelerator.prepare`
    # see https://gist.github.com/vwxyzjn/2581bff1e48e185e0b85b6dfe1def79c
    torch.manual_seed(args.seed)
    def repeat_generator():
        while True:
            yield from dataloader
    iter_dataloader = iter(repeat_generator())
    torch.manual_seed(local_seed)  # reset the local seed again

    if args.deepspeed:
        raise NotImplementedError("DeepSpeed is not supported.")
    else:
        reward_model = reward_model.to(device)

    kl_ctl = AdaptiveKLController(args.reward.kl_coef, hparams=args.reward.adaptive_kl)
    generation_config = GenerationConfig(
        max_new_tokens=args.task.response_length,
        min_new_tokens=args.task.response_length,
        temperature=(args.task.temperature + args.eps),
        top_k=0.0,
        top_p=1.0,
        do_sample=True,
    )
    # use the same `0.01` temperature for validation response generation https://github.com/openai/summarize-from-feedback/blob/700967448d10004279f138666442bf1497d0e705/exps/sample.py#L27
    validation_generation_config = GenerationConfig(
        max_new_tokens=args.task.response_length,
        min_new_tokens=args.task.response_length,
        temperature=(0.01 + args.eps),
        top_k=0.0,
        top_p=1.0,
        do_sample=True,
    )
    # Note: Don't add eos_token_id to the above generation configs - we don't want to avoid generating the eos token.

    # ========= Main Training Loop =========
    global_step = 0
    start_time = time.time()

    for update in range(1, args.ppo.num_updates + 1):
        global_step += 1 * args.batch_size
        with torch.no_grad():
            # ============ Gathering training samples ============
            data = next(iter_dataloader)
            # 1. Extract queries and reference responses from the dataset
            instructions = data["instruction"]
            queries = maybe_use_chat_template(instructions,
                                            use_chat_template = args.use_chat_template,
                                            tokenizer = tokenizer,
                                            args = args,
                                            device = device,
                                            )
            context_length = queries.shape[1]
            if args.use_chat_template:
                assert context_length == args.task.query_length + args.task.chat_template_buffer_length, f"Context length {context_length} does not match query length {args.task.query_length + args.task.chat_template_buffer_length}"
            else:
                assert context_length == args.task.query_length, f"Context length {context_length} does not match query length {args.task.query_length}"

            # 2. Generate responses using the given policy model
            total_sequence_length = args.task.query_length + args.task.chat_template_buffer_length + args.task.response_length
            if args.use_chat_template:
                context_length = args.task.query_length + args.task.chat_template_buffer_length
                vllm_tokenizer = chat_template_tokenizer
            else:
                context_length = args.task.query_length
                vllm_tokenizer = tokenizer

            # Decode tensor queries to strings for VLLM
            # Use skip_special_tokens=False initially, VLLM might need them depending on model
            prompt_strings = vllm_tokenizer.batch_decode(queries, skip_special_tokens=False)
            # Remove padding tokens - VLLM handles padding internally
            cleaned_prompts = [prompt.replace(vllm_tokenizer.pad_token, "").strip() for prompt in prompt_strings]

            LORA_DIR = "models/temp_lora"
            policy.save_pretrained(LORA_DIR)
            del policy
            with open(os.path.join(LORA_DIR, "args.pkl"), "wb") as f:
                pkl.dump(args, f)
            with open(os.path.join(LORA_DIR, "prompts.pkl"), "wb") as f:
                pkl.dump(cleaned_prompts, f)
            with open(os.path.join(LORA_DIR, "generation_config.pkl"), "wb") as f:
                pkl.dump(generation_config, f)
            vllm_tokenizer.save_pretrained(LORA_DIR)
            process = subprocess.Popen(["python", "src/dips/ultrafeedback/big_batch/generate.py", 
                                        "--base_model", args.base_model, 
                                        "--lora_dir", LORA_DIR,
                                        "--output_length", str(total_sequence_length), 
                                        "--context_length", str(context_length), 
                                        "--device", "cuda", 
                                        "--save_dir", os.path.join(LORA_DIR, "output"),
                                        "--args_file", os.path.join(LORA_DIR, "args.pkl"),
                                        "--prompts_file", os.path.join(LORA_DIR, "prompts.pkl"),
                                        "--generation_config_file", os.path.join(LORA_DIR, "generation_config.pkl"),
                                        "--n_outputs_per_prompt", str(args.rloo_k)
                                        ])
            process.wait()
            response_tensor = torch.load(os.path.join(LORA_DIR, "output", "responses.pkl")).to(device)

            # Truncate response after the first occurrence of `truncate_token_id`
            if args.swap_eos_token:
                postprocessed_response = truncate_response(args, chat_template_tokenizer, response_tensor)
                logprob_mask = postprocessed_response == chat_template_tokenizer.eos_token_id
            else:
                postprocessed_response = truncate_response(args, tokenizer, response_tensor)
                logprob_mask = postprocessed_response == tokenizer.pad_token_id
            sequence_length = first_true_indices(logprob_mask) - 1

            repeat_interleave_queries = torch.repeat_interleave(queries, args.rloo_k, dim = 0)
            query_response_tensor = torch.cat([repeat_interleave_queries, response_tensor], dim = 1)

            torch.save(response_tensor, os.path.join(LORA_DIR, "output", "response_tensor.pkl"))
            torch.save(query_response_tensor, os.path.join(LORA_DIR, "output", "query_response_tensor.pkl"))
            torch.save(logprob_mask, os.path.join(LORA_DIR, "output", "logprob_mask.pkl"))
            torch.save(queries, os.path.join(LORA_DIR, "output", "queries.pkl"))
            torch.save(sequence_length, os.path.join(LORA_DIR, "output", "sequence_length.pkl"))
            with open(os.path.join(LORA_DIR, "output", "args.pkl"), "wb") as f:
                pkl.dump(args, f)
            torch.save(postprocessed_response, os.path.join(LORA_DIR, "output", "postprocessed_response.pkl"))
            process = subprocess.Popen([".venv/bin/accelerate", "launch", "--num_processes", "1",
                                        "src/dips/ultrafeedback/big_batch/train.py", 
                                        "--lora_dir", LORA_DIR,
                                        "--response_tensor_file", os.path.join(LORA_DIR, "output", "response_tensor.pkl"),
                                        "--query_response_tensor_file", os.path.join(LORA_DIR, "output", "query_response_tensor.pkl"),
                                        "--logprob_mask_file", os.path.join(LORA_DIR, "output", "logprob_mask.pkl"),
                                        "--context_length", str(context_length),
                                        "--queries_file", os.path.join(LORA_DIR, "output", "queries.pkl"),
                                        "--sequence_length_file", os.path.join(LORA_DIR, "output", "sequence_length.pkl"),
                                        "--args_file", os.path.join(LORA_DIR, "output", "args.pkl"),
                                        "--postprocessed_response_file", os.path.join(LORA_DIR, "output", "postprocessed_response.pkl"),
                                        "--update_num", str(update)])
            process.wait()

            policy, _, _ = initialize_policy_with_optimizer(args, 
                                                            model_config, 
                                                            grad_norm_logger,
                                                            load_from_checkpoint = True,
                                                            lora_dir = args.output_dir)