# TODO: Port updates to kl.py over

import os
import time
from dataclasses import asdict
from types import SimpleNamespace
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
import tyro
import wandb
from accelerate import Accelerator
import accelerate
from torch.utils.tensorboard import SummaryWriter
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
from dips.ultrafeedback.big_batch.generate import generate_vllm, forward
from dips.ultrafeedback.big_batch.reward import get_reward

wandb.login(key=os.environ["WANDB_API_KEY"])

if __name__ == "__main__":

    # ========= Setup =========
    args = tyro.cli(Args)
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps) 

    local_seed = args.seed + accelerator.process_index * 100003  # Prime
    set_seed(local_seed)
    torch.backends.cudnn.deterministic = True

    args.world_size = accelerator.num_processes
    args.batch_size = args.per_device_train_batch_size * args.world_size

    if (args.local_rollout_forward_batch_size * args.rloo_k) % (args.gradient_accumulation_steps * args.per_device_train_batch_size) != 0:
        warnings.warn("local_rollout_forward_batch_size * rloo_k is not divisible by batch_size (gradient accumulation will require memory for the remainder)")

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

    # Add train type annotation to the experiment name
    if args.train_dips:
        final_exp_name = f"{args.exp_name}_dips"
        algo_subdir = "dips"
        final_output_dir = os.path.join(args.output_dir, algo_subdir)
    else:
        final_exp_name = f"{args.exp_name}_rloo"
        algo_subdir = "rloo"
        final_output_dir = os.path.join(args.output_dir, algo_subdir)
    os.makedirs(final_output_dir, exist_ok=True)

    run_name = f"{final_exp_name}__{args.seed}__{algo_subdir}"

    # ========= Logging =========
    console = Console(force_terminal=True)
    grad_norm_logger = GradNormLogger()
    if accelerator.is_main_process:
        if args.track:
            wandb.init(
                project=args.wandb_project_name,
                sync_tensorboard=True,
                config=asdict(args),
                name=run_name,
                save_code=True,
            )
            file_extensions = [".toml", ".lock", ".py", ".sh", ".yaml"]
            wandb.run.log_code(".", include_fn=lambda path: any([path.endswith(ext) for ext in file_extensions]))
        else:
            wandb.init(mode="disabled")
        writer = SummaryWriter(f"runs/{run_name}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )
        pprint(args)
    else:
        writer = SimpleNamespace()  # dummy writer
        writer.add_scalar = lambda x, y, z: None
        writer.add_histogram = lambda x, y, z: None

    device = accelerator.device

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
    if accelerator.is_main_process:
        pprint(model_config)
        pprint(reward_model.config)

    policy, optimizer, scheduler = initialize_policy_with_optimizer(args, model_config, accelerator, grad_norm_logger)

    # ========= Data =========
    dataloader, validation_dataloader = get_dataloaders(args, tokenizer)

    # sync random states for DataLoader(shuffle=True) before `accelerator.prepare`
    # see https://gist.github.com/vwxyzjn/2581bff1e48e185e0b85b6dfe1def79c
    torch.manual_seed(args.seed)
    model, optimizer, dataloader = accelerator.prepare(policy, optimizer, dataloader)
    validation_dataloader = accelerator.prepare(validation_dataloader)
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

    model.train()
    for update in trange(1, args.ppo.num_updates + 1):
        global_step += 1 * args.batch_size
        frac = 1.0 - ((update - 1.0) / args.ppo.num_updates)
        lrnow = frac * args.lr # linear learning rate decay
        optimizer.param_groups[0]["lr"] = lrnow
        with torch.no_grad():
            eval_storage, eval_df = evaluate(
                args = args,
                reward_model = reward_model,
                policy = accelerator.unwrap_model(model),
                tokenizer = tokenizer,
                dataloader = validation_dataloader,
                generation_config = validation_generation_config,
                accelerator = accelerator,
            )
            validation_score = eval_storage.score[0]
            if args.print_sample_output_freq > 0 and update > 1 and (update - 1) % args.print_sample_output_freq == 0:
                if accelerator.is_main_process:
                    eval_df.to_csv(f"runs/{run_name}/table_{global_step}.csv")
                    if args.track:
                       wandb.log({f"samples/query_responses_{update}": wandb.Table(dataframe=eval_df)}, step=update)
                    else:
                       try:
                           print_rich_table(f"Sample Output at Step {update}", eval_df[:1], console)
                       except Exception as e:
                           print(e)
                if args.run_eval:
                    eval_storage, eval_df = evaluate(
                        args,
                        reward_model,
                        accelerator.unwrap_model(model),
                        tokenizer,
                        validation_dataloader,
                        validation_generation_config,
                        sampling=False,
                        accelerator = accelerator,
                    )
                    if accelerator.is_main_process:
                        eval_df.to_csv(f"runs/{run_name}/table.csv")
                        if args.track:
                            wandb.log({f"eval/query_responses_{update}": wandb.Table(dataframe=eval_df)}, step=update)

                # save model
                if args.output_dir:
                    os.makedirs(os.path.dirname(args.output_dir), exist_ok=True)
                    time_tensor = torch.tensor([int(time.time())], device=device)
                    time_int = accelerator.gather(time_tensor)[0].item()  # avoid different timestamps across processes
                    repo_name = f"{args.base_model.replace('/', '_')}__{args.exp_name}__tldr"
                    repo_id = f"{args.hf_entity}/{repo_name}" if args.hf_entity else repo_name

                    if accelerator.is_main_process:
                        tokenizer.save_pretrained(args.output_dir, repo_id=repo_id)
                        if args.push_to_hub:
                            tokenizer.push_to_hub(repo_id, revision=f"seed{args.seed}_{str(time_int)}")

                    unwrapped: PreTrainedModel = accelerator.unwrap_model(model)
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        unwrapped.save_pretrained(
                            args.output_dir,
                            is_main_process=accelerator.is_main_process,
                            save_function=accelerator.save,
                            state_dict=accelerator.get_state_dict(unwrapped),
                            safe_serialization=False,
                            repo_id=repo_id,
                        )
                        if args.push_to_hub:
                            unwrapped.push_to_hub(repo_id, revision=f"seed{args.seed}_{str(time_int)}", safe_serialization=False)
            del eval_storage, eval_df
            torch.cuda.empty_cache()

            # ============ Gathering training samples ============
            data = next(iter_dataloader)
            instructions = data["instruction"]
            queries = maybe_use_chat_template(instructions,
                                            use_chat_template = args.use_chat_template,
                                            tokenizer = tokenizer,
                                            args = args,
                                            device = device,
                                            )

            # 1. Extract queries and reference responses from the dataset
            instruction = data["instruction"]
            queries = maybe_use_chat_template(instruction, 
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
            if accelerator.is_main_process:
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

                model.to("cpu")
                response_tensor = generate_vllm(
                    policy = accelerator.unwrap_model(model),
                    accelerator = accelerator,
                    prompts = cleaned_prompts,
                    tokenizer = vllm_tokenizer,
                    generation_config = generation_config,
                    context_length = context_length,
                    output_length = total_sequence_length,
                    device = device,
                    args = args,
                    n_outputs_per_prompt = args.rloo_k,
                )
                model.to(device)
                for param in model.parameters():
                    param.requires_grad = True
                del vllm_tokenizer
                
            accelerator.wait_for_everyone()
            # Broadcast the generated responses from the main process to all other processes
            response_tensor = accelerate.utils.broadcast(response_tensor, from_process=0)
            accelerator.wait_for_everyone()

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

            # 3. Calculate the logprobs and ref_logprobs for the given responses
            logprobs = []
            ref_logprobs = []
            scores = []
            for start_idx in range(0, len(query_response_tensor), args.local_rollout_forward_batch_size):
                end_idx = min(start_idx + args.local_rollout_forward_batch_size, len(query_response_tensor))
                query_response_batch = query_response_tensor[start_idx:end_idx]
                logprob_mask_batch = logprob_mask[start_idx:end_idx]

                response_batch = query_response_batch[:, context_length:]
                response_batch = torch.masked_fill(response_batch, logprob_mask_batch, 0)

                output = forward(model = accelerator.unwrap_model(model), 
                                 input_ids = query_response_batch, 
                                 tokenizer = tokenizer,
                                 args = args)
                if accelerator.is_main_process and accelerator.is_local_main_process and args.track:
                    wandb.config.update({"model/output_dtype": str(output.logits.dtype)})

                logits = output.logits[:, context_length - 1 : -1]
                logits /= (args.task.temperature + args.eps)
                all_logprob = F.log_softmax(logits, dim=-1)
                logprob = torch.gather(all_logprob, 2, response_batch.unsqueeze(-1)).squeeze(-1)

                # Mask out padding tokens (we don't want to calculate KL divergence on them)
                logprob = torch.masked_fill(logprob, logprob_mask_batch, 0)
                del output, logits, all_logprob

                ref_output = forward(model = accelerator.unwrap_model(model), 
                                     input_ids = query_response_batch, 
                                     tokenizer = tokenizer,
                                     args = args,
                                     ref = True)
                ref_logits = ref_output.logits[:, context_length - 1 : -1]
                # Pad output sequence to response_length (necessary to avoid shape mismatch across devices)
                # pad = torch.ones(ref_logits.shape[0], args.task.response_length-ref_logits.shape[1], dtype = ref_logits.dtype).to(device)
                # ref_logits = torch.cat([ref_logits, pad], dim = 1)
                ref_logits /= (args.task.temperature + args.eps)
                ref_all_logprob = F.log_softmax(ref_logits, dim=-1)
                ref_logprob = torch.gather(ref_all_logprob, 2, response_batch.unsqueeze(-1)).squeeze(-1)

                ref_logprob = torch.masked_fill(ref_logprob, logprob_mask_batch, 0)
                del ref_output, ref_logits, ref_all_logprob
                torch.cuda.empty_cache()

                # # Response Processing 2. run reward model on the truncated responses
                # repeated_instructions = []
                # for inst in instruction_batch:
                #     repeated_instructions.extend([inst] * args.rloo_k) # effectively torch.repeat_interleave
                score, reward_breakdown, reward_breakdown_coeffs = get_reward(reward_model = reward_model, 
                                input_ids = query_response_batch,
                                tokenizer = tokenizer)

                logprobs.append(logprob)
                ref_logprobs.append(ref_logprob)
                scores.append(score)

                torch.cuda.empty_cache()

            scores = torch.cat(scores, 0)
            logprobs = torch.cat(logprobs, 0)
            ref_logprobs = torch.cat(ref_logprobs, 0)

            torch.cuda.empty_cache()

            # scale RM scores
            scores = scores * args.task.reward_coef
            assert scores.shape == torch.Size([queries.shape[0] * args.rloo_k]), f"scores.shape {scores.shape} does not match queries.shape {queries.shape} * args.rloo_k {args.rloo_k}"

            # Response Processing 3. filter response. Ensure that the sample contains truncate_token_id (doesn't exceed max len)
            # responses not passing that filter will receive a low (fixed) score
            # only query RM on responses that pass that filter
            if args.swap_eos_token:
                contain_eos_token = torch.any(response_tensor == chat_template_tokenizer.eos_token_id, dim=-1)
            else:
                contain_eos_token = torch.any(response_tensor == tokenizer.eos_token_id, dim=-1)
            scores = torch.where(contain_eos_token, scores, torch.full_like(scores, args.task.penalty_reward_value))
            penalty_frac = 1 - (contain_eos_token.sum() / len(contain_eos_token))

            # Calculate baselines
            if args.rloo_k > 1:
                # The shape of score is [batch_size * rloo_k]
                per_prompt_scores = scores.reshape(-1, args.rloo_k)
                per_prompt_logprobs = torch.sum(logprobs, axis = 1).reshape(-1, args.rloo_k)
                per_prompt_ref_logprobs = torch.sum(ref_logprobs, axis = 1).reshape(-1, args.rloo_k)
                per_prompt_approx_kl  = per_prompt_logprobs - per_prompt_ref_logprobs
                kl_baseline = (per_prompt_approx_kl.sum(dim = 1, keepdim = True) - per_prompt_approx_kl) / (args.rloo_k - 1)
                score_baseline = (per_prompt_scores.sum(dim = 1, keepdim = True) - per_prompt_scores) / (args.rloo_k - 1)
                baselines = score_baseline - kl_ctl.value * kl_baseline
                baselines = baselines.reshape(-1)
            else:
                baselines = torch.zeros_like(scores)

            # 4. compute rewards
            kl = logprobs - ref_logprobs # [batch_size, response_len]
            non_score_reward = -kl_ctl.value * kl
            rewards = non_score_reward.clone()
            actual_start = torch.arange(rewards.size(0), device=rewards.device)
            actual_end = sequence_length
            rewards[[actual_start, actual_end]] += scores
            writer.add_scalar("generation/seq_len_mean", sequence_length.to(torch.float32).mean().item(), update)
            writer.add_scalar("generation/seq_len_std", sequence_length.to(torch.float32).std().item(), update)
            writer.add_scalar("generation/seq_len_max", sequence_length.max().item(), update)
            writer.add_scalar("generation/seq_len_min", sequence_length.min().item(), update)

            # Log reward breakdowns to wandb
            reward_breakdown_dict, reward_breakdown_coeffs_dict = parse_reward_breakdown_attributes(reward_breakdown = reward_breakdown, 
                                                                                reward_breakdown_coeffs = reward_breakdown_coeffs)
            if args.track:
                for key in reward_breakdown_dict:
                    writer.add_scalar(f"reward_breakdown/{key}", reward_breakdown_dict[key], update)
                    writer.add_scalar(f"reward_breakdown_coeffs/{key}", reward_breakdown_coeffs_dict[key], update)
            del reward_breakdown, reward_breakdown_coeffs, reward_breakdown_dict, reward_breakdown_coeffs_dict

            torch.cuda.empty_cache()

        # center = 0.1 * torch.mean(torch.sum(non_score_reward, axis=1) + scores)
        # Do multiple epochs of PPO training, with a fresh random shuffle in each epoch

        stats_shape = (args.ppo.noptepochs)
        metrics = defaultdict(lambda: torch.zeros(stats_shape, device = device))
        num_samples = len(query_response_tensor)
        num_minibatches = num_samples // args.per_device_train_batch_size

        if args.use_chat_template:
            context_length = args.task.query_length + args.task.chat_template_buffer_length
        else:
            context_length = args.task.query_length
        
        model.train()
        for ppo_epoch_idx in range(args.ppo.noptepochs):
            local_batch_idxs = np.random.permutation(num_samples)
            for mini_batch_start in range(0, num_samples, args.per_device_train_batch_size):
                mini_batch_end = mini_batch_start + args.per_device_train_batch_size
                mini_batch_inds = local_batch_idxs[mini_batch_start:mini_batch_end]
                with accelerator.accumulate(model):
                    # These are all fixed and won't get gradients
                    mb_responses = response_tensor[mini_batch_inds] # [batch_size, response_len]
                    mb_query_responses = query_response_tensor[mini_batch_inds] # [batch_size, seq_len]
                    mb_postprocessed_responses = postprocessed_response[mini_batch_inds] # [batch_size, response_len]
                    mb_logprobs = torch.sum(logprobs[mini_batch_inds], axis=1) # [batch_size]
                    mb_ref_logprobs = torch.sum(ref_logprobs[mini_batch_inds], axis=1)
                    mb_reward = scores[mini_batch_inds]
                    mb_baseline = baselines[mini_batch_inds]

                    # compute the logprobs w/ gradient tracking
                    output = forward(model = accelerator.unwrap_model(model), 
                                     input_ids = mb_query_responses.clone().detach(), 
                                     tokenizer = tokenizer,
                                     args = args)
                    # output.logits has shape [batch_size, seq_len, vocab_size]
                    logits = output.logits[:, context_length - 1 : -1] # logits of response [batch_size, response_len, vocab_size]
                    # pad = torch.ones(logits.shape[0], args.task.response_length-logits.shape[1], logits.shape[2], dtype = logits.dtype).to(device)
                    # logits = torch.cat([logits, pad], dim = 1)
                    logits /= (args.task.temperature + args.eps)
                    new_all_logprobs = F.log_softmax(logits, dim=-1) # [batch_size, response_len, vocab_size]

                    if args.swap_eos_token:
                        logprob_mask = mb_postprocessed_responses == chat_template_tokenizer.eos_token_id
                    else:
                        logprob_mask = mb_postprocessed_responses == tokenizer.pad_token_id
                    mb_responses_no_padding = torch.masked_fill(mb_responses, logprob_mask, 0)
                    # index logprobs over vocab dim by what the model actually generated
                    new_logprobs = torch.gather(new_all_logprobs, 2, mb_responses_no_padding.unsqueeze(-1)).squeeze(-1)
                    # shape [batch_size] (total logprob of the response)
                    new_logprobs = torch.masked_fill(new_logprobs, logprob_mask, 0)
                    new_logprobs = torch.sum(new_logprobs, axis=1)
                    with torch.amp.autocast(device_type = "cuda",
                                            enabled = not args.loss_full_precision):
                        if args.train_dips:
                            # the IPS trick loss
                            approx_kl = new_logprobs - mb_ref_logprobs
                            prob_ratio = torch.exp(new_logprobs - mb_logprobs)
                            weighting = (mb_reward - mb_baseline - kl_ctl.value * approx_kl)

                            if args.factor_loss:
                                policy_loss_term = (prob_ratio * weighting.detach()).mean()
                                kl_loss_term = (prob_ratio.detach() * weighting).mean()
                                loss = -1 * (policy_loss_term + kl_loss_term)
                            else:
                                loss = torch.mean(-1 * prob_ratio * weighting)

                        else:
                            # RLOO loss
                            approx_kl = mb_logprobs - mb_ref_logprobs
                            weighting = (mb_reward - mb_baseline - kl_ctl.value * approx_kl)
                            loss = torch.mean(-1*new_logprobs * weighting)
                            if args.kl_grad_patch:
                                differentiable_kl = new_logprobs - mb_ref_logprobs
                                diff_reward = (mb_reward - mb_baseline - kl_ctl.value * differentiable_kl)
                                loss = loss + torch.mean(-1 * diff_reward)

                    # Grab model grad norms
                    # if args.train_dips and args.factor_loss:
                    #     policy_term_grad_norms = grad_norm_logger.get_grad_norms(loss = policy_loss_term)
                    #     kl_term_grad_norms = grad_norm_logger.get_grad_norms(loss = kl_loss_term)

                    # grad_norms = grad_norm_logger.get_grad_norms(loss = loss)

                    accelerator.backward(loss)
                    if args.clip_grad_norm is not None and accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(model.parameters(), args.clip_grad_norm)

                    optimizer.step()
                    optimizer.zero_grad()

                with torch.no_grad():
                    # Do whatever logging we want
                    metrics["loss"][ppo_epoch_idx] += loss.detach().mean()
                    metrics["baseline"][ppo_epoch_idx] += mb_baseline.mean()
                    # metrics["grad_norm_mean"][ppo_epoch_idx] += grad_norms.mean()
                    # metrics["grad_norm_max"][ppo_epoch_idx] += grad_norms.max()
                    # metrics["grad_norm_std"][ppo_epoch_idx] += grad_norms.std()
                    metrics["penalty_frac"][ppo_epoch_idx] += penalty_frac.mean().item()

                    if args.train_dips:
                        metrics["weighting"][ppo_epoch_idx] += weighting.mean()
                        metrics["prob_ratio"][ppo_epoch_idx] += prob_ratio.mean()
                        metrics["approx_kl"][ppo_epoch_idx] += approx_kl.mean()
                        # if args.factor_loss:
                        #     # No need to log kl and policy grad norms - they're both the same as the loss.
                        #     # The distinction is in the gradient flow.
                        #     metrics["policy_grad_norm_mean"][ppo_epoch_idx] += policy_term_grad_norms.mean()
                        #     metrics["policy_grad_norm_max"][ppo_epoch_idx] += policy_term_grad_norms.max()
                        #     metrics["policy_grad_norm_std"][ppo_epoch_idx] += policy_term_grad_norms.std()
                        #     metrics["kl_grad_norm_mean"][ppo_epoch_idx] += kl_term_grad_norms.mean()
                        #     metrics["kl_grad_norm_max"][ppo_epoch_idx] += kl_term_grad_norms.max()
                        #     metrics["kl_grad_norm_std"][ppo_epoch_idx] += kl_term_grad_norms.std()
                    else:
                        metrics["weighting"][ppo_epoch_idx] += weighting.mean()
                        metrics["new_logprobs"][ppo_epoch_idx] += new_logprobs.mean()
                        metrics["approx_kl"][ppo_epoch_idx] += approx_kl.mean()

                    
        with torch.no_grad():
            mean_kl = kl.sum(1).mean()
            mean_entropy = (-logprobs).sum(1).mean()
            mean_non_score_reward = non_score_reward.sum(1).mean()

            writer.add_scalar("objective/kl_coef", kl_ctl.value, update)
            writer.add_scalar("objective/kl", accelerator.gather(mean_kl).mean().item(), update)
            writer.add_scalar("objective/entropy", accelerator.gather(mean_entropy).mean().item(), update)
            writer.add_scalar("objective/non_score_reward", accelerator.gather(mean_non_score_reward).mean().item(), update)
            writer.add_scalar(
                "objective/score_total", accelerator.gather(mean_non_score_reward + scores.mean()).mean().item(), update
            )
            writer.add_scalar("objective/scores", accelerator.gather(scores.mean()).mean().item(), update)
            writer.add_scalar("objective/validation_score", accelerator.gather(validation_score.mean()).mean().item(), update)

            writer.add_scalar("train/reward", accelerator.gather(scores.mean()).mean().item(), update)
            writer.add_scalar("train/reward_std", accelerator.gather(scores).std().item(), update)
            writer.add_scalar("train/kl", accelerator.gather(mean_kl).mean().item(), update)

            for stats in metrics:
                writer.add_scalar(f"train/{stats}", accelerator.gather(metrics[stats]).mean().item() / num_minibatches, update)

            scheduler.step()
            writer.add_scalar("train/lr", scheduler.get_last_lr()[0], update)

            if args.reward.use_adaptive_kl:
                kl_ctl.update(mean_kl.item(), args.batch_size)
            
            del output, logits, new_all_logprobs, new_logprobs, approx_kl, weighting, loss #, grad_norms
            del kl, mean_kl, mean_entropy, mean_non_score_reward, scores

            torch.cuda.empty_cache()
            if args.force_clear_grad_optim:
                if (args.local_rollout_forward_batch_size * args.rloo_k) % (args.gradient_accumulation_steps * args.per_device_train_batch_size * args.world_size) == 0:
                    force_clear_grads(accelerator, model, optimizer) # Note: We want to pass in the model instead of accelerator.unwrap(model) to access the _no_sync_context.

    if args.run_eval:
        eval_storage, eval_df = evaluate(
            args,
            reward_model,
            accelerator.unwrap_model(model),
            tokenizer,
            validation_dataloader,
            validation_generation_config,
            sampling=False,
            max_eval_size = args.max_eval_size,
        )
        if accelerator.is_main_process:
            eval_df.to_csv(f"runs/{run_name}/table.csv")
            if args.track:
                wandb.log({"eval/query_responses": wandb.Table(dataframe=eval_df)}, step=update)

    # save model
    if args.output_dir:
        os.makedirs(os.path.dirname(args.output_dir), exist_ok=True)
        time_tensor = torch.tensor([int(time.time())], device=device)
        time_int = accelerator.gather(time_tensor)[0].item()  # avoid different timestamps across processes
        repo_name = f"{args.base_model.replace('/', '_')}__{args.exp_name}__tldr"
        repo_id = f"{args.hf_entity}/{repo_name}" if args.hf_entity else repo_name

        if accelerator.is_main_process:
            tokenizer.save_pretrained(args.output_dir, repo_id=repo_id)
            if args.push_to_hub:
                tokenizer.push_to_hub(repo_id, revision=f"seed{args.seed}_{str(time_int)}")

        unwrapped: PreTrainedModel = accelerator.unwrap_model(model)
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            unwrapped.save_pretrained(
                args.output_dir,
                is_main_process=accelerator.is_main_process,
                save_function=accelerator.save,
                state_dict=accelerator.get_state_dict(unwrapped),
                safe_serialization=False,
                repo_id=repo_id,
            )
            if args.push_to_hub:
                unwrapped.push_to_hub(repo_id, revision=f"seed{args.seed}_{str(time_int)}", safe_serialization=False)
