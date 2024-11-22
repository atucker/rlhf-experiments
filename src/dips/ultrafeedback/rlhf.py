import os
import random
import time
from dataclasses import asdict, dataclass, field
from types import SimpleNamespace
from typing import List, Literal, Optional, Tuple, Union
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
import wandb
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.state import AcceleratorState
from accelerate.utils import gather_object
from datasets import load_dataset
from rich.console import Console
from rich.pretty import pprint
from rich.table import Table
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    GenerationConfig,
    PretrainedConfig,
    PreTrainedModel,
)
from peft import get_peft_model, LoraConfig
import random

# Package imports
from dips.utils import set_seed, get_grad_norms

wandb.login(key=os.environ["WANDB_API_KEY"])

@dataclass
class AdaptiveKLParams:
    target: float = 6.0
    horizon: int = 10000  # in episodes

@dataclass
class RewardHParams:
    use_adaptive_kl: bool = False
    adaptive_kl: Optional[AdaptiveKLParams] = field(default_factory=AdaptiveKLParams)
    kl_coef: float = 0.05

@dataclass
class PpoHParams:
    num_updates: tyro.conf.Suppress[int] = None
    noptepochs: int = 1 # number of epochs to train on each PPO update
    vf_coef: float = 0.1
    cliprange: float = 0.2
    cliprange_value: float = 0.2
    gamma: float = 1
    lam: float = 0.95
    whiten_rewards: bool = False


@dataclass
class TaskHParams:
    # Query params
    query_length: int = 512
    query_dataset: str = "GitBag/ultrafeedback_llama3_eurus"

    # Response params
    response_length: int = 128

    # Truncate response after the first occurrence of this token at or after index after when sampling.
    truncate_token: Literal["eos"] = "eos"
    truncate_token_id: Optional[int] = None
    penalty_reward_value: int = -1

    # LM params
    temperature: float = 0.5


@dataclass
class Args:
    train_dips: bool = True # whether to train via DIPS or RLOO
    disable_wandb: bool = True
    # common args
    exp_name: str = "llama_3_8b_ultrafeedback"
    """the name of this experiment"""
    seed: int = 55134
    """seed of the experiment"""
    track: bool = True
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "llama_3_8b_ultrafeedback"
    """the wandb's project name"""
    cuda: bool = True
    """Whether to use cuda if available."""
    run_name: Optional[str] = None
    """a unique name of this run"""
    push_to_hub: bool = False
    "whether to upload the saved model to huggingface"
    hf_entity: str = ""
    "the user or org name of the model repository from the Hugging Face Hub"
    deepspeed: bool = False
    """Whether to use deepspeed to train the model"""
    print_sample_output_freq: int = 1000
    """How often to print sample output"""
    run_eval: bool = True
    """Whether to run evaluation"""

    # optimizer args
    eps: float = 1e-5
    """the epsilon value for the optimizer - an extremely small value to prevent division by zero"""
    lr: float = 1e-7
    """the learning rate"""
    optimizer: Literal["adam", "adamw"] = "adamw"
    """Which optimizer to use"""
    scheduler: str = "linear"
    """Which scheduler to use"""
    warm_up_steps: int = 0
    """Number of warm up steps for the scheduler"""

    gradient_accumulation_steps: int = 1
    """The number of gradient accumulation steps"""

    # ------ Batch Size in Memory / GPU: per_device_train_batch_size --------
    rloo_k: int = 1 # number of samples to use for RLOO
    
    per_device_train_batch_size: int = 1
    """The micro batch size per GPU (HF's `per_device_train_batch_size`)"""
    per_device_eval_batch_size: int = 1
    """per rank eval batch size"""
    local_rollout_forward_batch_size: int = 1
    """per rank no grad forward pass in the rollout phase"""

    total_episodes: int = int(1e6) # Informs the number of ppo updates to do
    """The total number of episodes in the dataset"""

    # optional args filled while running
    world_size: Optional[int] = 2
    """The number of processes (GPUs) to use"""

    # other args
    base_model: str = "neuralmagic/Meta-Llama-3-8B-Instruct-quantized.w8a16"
    """the name of the pretrained model to use"""
    offload: bool = False
    """Whether to offload ref policy and reward model to CPU"""
    reward_model_path: str = "RLHFlow/ArmoRM-Llama3-8B-v0.1"
    """the name of the pretrained model to use"""
    sft_model_path: str = "neuralmagic/Meta-Llama-3-8B-Instruct-quantized.w8a16"
    """the name of the pretrained model to use"""
    dropout_layer_keys: List[str] = field(
        default_factory=lambda: ["attn_pdrop", "embd_pdrop", "resid_pdrop", "summary_first_dropout"]
    )
    """Which layers to apply dropout to"""
    output_dir: str = "models/llama_3_8b_armoRM_ultrafeedback"
    """Where to save the model"""
    lora_rank: int = 4
    """the rank of the lora matrix"""
    lora_alpha: int = 8
    """weight of lora"""
    lora_dropout: float = 0.0
    """dropout for lora"""
    task: TaskHParams = field(default_factory=TaskHParams)
    reward: RewardHParams = field(default_factory=RewardHParams)
    ppo: PpoHParams = field(default_factory=PpoHParams)


# taken from https://github.com/microsoft/DeepSpeedExamples/blob/737c6740bec38b77a24a59135b6481a53d566b38/applications/DeepSpeed-Chat/training/utils/model/model_utils.py#L20C1-L26C52
def configure_dropout(model_config, dropout_layer_keys, dropout):
    if dropout is not None:
        for key in dropout_layer_keys:
            if hasattr(model_config, key):
                print(f"Setting model_config.{key} to {dropout}")
                setattr(model_config, key, dropout)


def print_rich_table(title: str, df: pd.DataFrame, console: Console) -> Table:
    table = Table(show_lines=True)
    for column in df.columns:
        table.add_column(column)
    for _, row in df.iterrows():
        table.add_row(*row.astype(str).tolist())
    console.rule(f"[bold red]{title}")
    console.print(table)

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.normal_(layer.weight, std=std)
    torch.nn.init.constant_(layer.bias, val=bias_const)
    return layer


class AdaptiveKLController:
    def __init__(self, init_kl_coef: float, hparams: AdaptiveKLParams):
        self.value = init_kl_coef
        self.hparams = hparams

    def update(self, current, n_steps):
        target = self.hparams.target
        proportional_error = np.clip(current / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.hparams.horizon
        self.value *= mult


def whiten(values, shift_mean=True):
    # `unbiased=False` matches TF `tf.nn.moments`'s setting
    # Normalize the values to have a mean of 0 (if shift_mean is false) and a variance of 1
    mean, var = torch.mean(values), torch.var(values, unbiased=False)
    whitened = (values - mean) * torch.rsqrt(var + 1e-8)
    if not shift_mean:
        whitened += mean
    return whitened


class ScalarModelConfig(PretrainedConfig):
    def __init__(
        self,
        base_model: str = None,
        base_config: PretrainedConfig = None,
        hidden_size: int = 768,
        bias: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.base_model = base_model
        self.base_config = base_config
        self.hidden_size = hidden_size
        self.bias = bias


class ScalarModel(PreTrainedModel):
    config_class = ScalarModelConfig

    def __init__(self, config: ScalarModelConfig):
        super().__init__(config)
        self.config = config
        self.lm_backbone = AutoModel.from_pretrained(
            config.base_model,
            config=self.config.base_config,
            trust_remote_code=True,
        )
        self.scalar_head = layer_init(
            nn.Linear(self.config.hidden_size, 1),
            std=1 / np.sqrt(self.config.hidden_size + 1),
        )

    def forward(self, **kwargs):
        output = self.lm_backbone(**kwargs)
        reward = self.scalar_head(output.hidden_states[-1]) - self.config.bias
        return reward


def get_reward(reward_model: nn.Module, 
               instruction_str: List[str],
               response_str: List[str], 
               rm_tokenizer: AutoTokenizer) -> torch.Tensor:
    """
    Uses the reward model to calculate reward information for the given query_responses.

    Returns a scalar reward for each query_response pair.
    """
    messages = [[{"role": "user", "content": instruction},
                 {"role": "assistant", "content": response}] for instruction, response in zip(instruction_str, response_str)]
    input_ids = rm_tokenizer.apply_chat_template(messages, return_tensors="pt", padding = True)
    input_ids = torch.tensor(input_ids).to(device)
    attention_mask = input_ids != tokenizer.pad_token_id
    with torch.no_grad():
        output = reward_model(input_ids=input_ids, 
                              attention_mask=attention_mask, 
                              return_dict=True)

    # https://github.com/huggingface/transformers/blob/dc68a39c8111217683bf49a4912d0c9018bab33d/src/transformers/models/gpt2/modeling_gpt2.py#L1454
    return output.score


def exact_div(a, b):
    q = a // b
    if a != q * b:
        raise ValueError(f"Inexact division: {a} / {b} = {a / b}")
    return q


def generate(lm_backbone: AutoModelForCausalLM, 
             queries: torch.Tensor, 
             tokenizer: AutoTokenizer, 
             generation_config: GenerationConfig,
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
    )
    expanded_queries = queries.repeat_interleave(n_outputs_per_prompt, dim=0) # [batch_size * n_outputs_per_prompt, seq_len]
    full_sequences = torch.cat((expanded_queries, output.sequences[:, context_length:]), dim=1)
    return full_sequences

def debug_tensor_info(tensor, name):
    print(f"{name}:")
    print(f"- Shape: {tensor.shape}")
    print(f"- Device: {tensor.device}")
    print(f"- Dtype: {tensor.dtype}")
    print(f"- Memory: {tensor.element_size() * tensor.nelement() / 1024 / 1024:.2f}MB")
    print(f"- Requires grad: {tensor.requires_grad}")

def first_true_indices(bools, dtype=torch.long) -> torch.Tensor:
    """
    Takes an N-dimensional bool tensor and returns an (N-1)-dimensional tensor of integers giving
    the position of the first True in each "row".

    Returns the length of the rows (bools.size(-1)) if no element is True in a given row.
    """
    row_len = bools.size(-1)
    zero_or_index = row_len * (~bools).type(dtype) + torch.arange(row_len, dtype=dtype, device=bools.device)
    return torch.min(zero_or_index, dim=-1).values


def truncate_response(args, tokenizer, responses):
    """
    Truncates responses after the first occurrence of the truncate token.
    """
    trunc_idxs = first_true_indices(responses == args.task.truncate_token_id).unsqueeze(-1)
    new_size = [1] * (len(responses.size()) - 1) + [args.task.response_length]
    idxs = torch.arange(args.task.response_length, device=responses.device).view(*new_size)
    postprocessed_responses = torch.masked_fill(responses, idxs > trunc_idxs, tokenizer.pad_token_id)
    return postprocessed_responses


def forward(model: AutoModelForCausalLM, 
            query_responses: torch.Tensor, 
            tokenizer: AutoTokenizer, 
            ref: bool = False):
    """
    Get model output for given query_responses. 
    If ref is True, the model's adapter module is disabled (e.g peft models are reverted to their base models).
    """
    attention_mask = query_responses != tokenizer.pad_token_id
    input_ids = torch.masked_fill(query_responses, ~attention_mask, 0)
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

@dataclass
class EvalStorage:
    """
    Stores evaluation results from the reward model.
    """
    query_token: List[str] = field(default_factory=list)
    postprocessed_response_token: List[str] = field(default_factory=list)
    reference_response_token: List[str] = field(default_factory=list)
    score: List[float] = field(default_factory=list)
    reference_score: List[float] = field(default_factory=list)

    query: List[str] = field(default_factory=list)
    postprocessed_response: List[str] = field(default_factory=list)
    reference_response: List[str] = field(default_factory=list)
    kl: List[float] = field(default_factory=list)
    baseline: List[float] = field(default_factory=list)


def evaluate(args: Args, reward_model: nn.Module, policy: nn.Module, tokenizer: AutoTokenizer,
             dataloader: DataLoader, generation_config: GenerationConfig, sampling=True) -> Tuple[EvalStorage, pd.DataFrame]:
    """
    Completes an episode rollout for the policy model and returns:

    - reference response and reference response performance
    - policy-generated response and policy-generated response performance
    - kl divergence between the policy and reference model
    """
    eval_storage = EvalStorage()
    with torch.no_grad():
        for data in dataloader:
            # 1. Extract queries and reference responses from the dataset
            queries = data["llama_prompt_tokens"].to(device)
            instruction = data["instruction"]
            context_length = queries.shape[1]

            # 2. Generate responses using the given policy model
            query_responses = generate(
                lm_backbone = accelerator.unwrap_model(policy),
                queries = queries,
                tokenizer = tokenizer,
                generation_config = generation_config,
                n_outputs_per_prompt = 1,
            ) # [batch_size * n_outputs_per_prompt, prompt_len + response_len]

            responses = query_responses[:, context_length:]

            postprocessed_responses = truncate_response(args, tokenizer, responses)
            # expanded_queries = queries.repeat_interleave(rloo_k, dim=0)
            # postprocessed_query_responses = torch.cat((queries, postprocessed_responses), 1)
            score = get_reward(reward_model = reward_model, 
                                     instruction_str = instruction,
                                     response_str = tokenizer.batch_decode(postprocessed_responses, skip_special_tokens=True),
                                     rm_tokenizer = rm_tokenizer)
            
            eval_storage.query_token.extend(queries)
            eval_storage.postprocessed_response_token.extend(postprocessed_responses)
            eval_storage.score.append(score)

            if sampling:
                break

    eval_storage.query = tokenizer.batch_decode(eval_storage.query_token, skip_special_tokens=True)
    eval_storage.postprocessed_response = tokenizer.batch_decode(
        eval_storage.postprocessed_response_token, skip_special_tokens=True
    )
    eval_score = torch.cat(eval_storage.score).float().cpu().numpy().tolist()
    eval_df = pd.DataFrame(
        {
            "query": gather_object(eval_storage.query),
            "postprocessed_response": gather_object(eval_storage.postprocessed_response),
            "scores": gather_object(eval_score),
        }
    )
    return eval_storage, eval_df

if __name__ == "__main__":

    args = tyro.cli(Args)
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps,
                              kwargs_handlers = [DistributedDataParallelKwargs(find_unused_parameters = True)]) 
    #                           ^ Necessary for the policy-value wrapper trick we pull
    local_seed = args.seed + accelerator.process_index * 100003  # Prime
    set_seed(local_seed)

    args.world_size = accelerator.num_processes
    args.batch_size = args.per_device_train_batch_size * args.world_size * args.gradient_accumulation_steps
    args.local_batch_size = args.per_device_train_batch_size * args.gradient_accumulation_steps
    if args.ppo.whiten_rewards:
        assert (
            args.local_batch_size >= 8
        ), f"Per-rank minibatch size {args.local_batch_size} is insufficient for whitening"
        # raise NotImplementedError("Whitening is not supported at the moment.")
    args.ppo.num_updates = args.total_episodes // args.batch_size

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        padding_side="right",
        trust_remote_code=True,
    )

    rm_tokenizer = AutoTokenizer.from_pretrained(
        args.reward_model_path,
        use_fast = True,
    )
    # we use the padding token manually but do not resize the token embedding of the model
    if args.task.truncate_token == "eos":
        args.task.truncate_token_id = tokenizer.eos_token_id
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    console = Console(force_terminal=True)

    # Add train type annotation to the experiment name
    if args.train_dips:
        args.exp_name = f"{args.exp_name}_dips"
        args.output_dir = f"{args.output_dir}_dips"
    else:
        args.exp_name = f"{args.exp_name}_rloo"
        args.output_dir = f"{args.output_dir}_rloo"
    run_name = f"{args.exp_name}__{args.seed}__{args.output_dir.split('/')[1]}"
    writer = SimpleNamespace()  # dummy writer
    writer.add_scalar = lambda x, y, z: None
    writer.add_histogram = lambda x, y, z: None
    if accelerator.is_main_process:
        if args.track:
            wandb.init(
                project=args.wandb_project_name,
                sync_tensorboard=True,
                config=asdict(args),
                name=run_name,
                save_code=True,
                mode="disabled" if args.disable_wandb else None,
            )
            file_extensions = [".toml", ".lock", ".py", ".sh", ".yaml"]
            wandb.run.log_code(".", include_fn=lambda path: any([path.endswith(ext) for ext in file_extensions]))
        writer = SummaryWriter(f"runs/{run_name}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )
        pprint(args)
    device = accelerator.device
    torch.backends.cudnn.deterministic = True

    model_config = AutoConfig.from_pretrained(args.base_model)
    configure_dropout(model_config, args.dropout_layer_keys, 0.0)  # disable dropout
    assert args.reward_model_path, "reward_model_path must be provided"
    reward_model: PreTrainedModel = AutoModelForSequenceClassification.from_pretrained(
        args.reward_model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    if accelerator.is_main_process:
        pprint(model_config)
        pprint(reward_model.config)

    policy = AutoModelForCausalLM.from_pretrained(args.sft_model_path, 
                                                  config=model_config, 
                                                  trust_remote_code=True,
                                                  torch_dtype="auto",
                                                  low_cpu_mem_usage = True)
    

    peft_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
    )
    for param in policy.parameters():
        param.requires_grad = False

    policy = get_peft_model(policy, peft_config=peft_config)
    param_subset = list(policy.parameters())
    accelerator.print(policy)
    policy.generation_config.eos_token_id = None  # disable `pad_token_id` and `eos_token_id` because we just want to
    policy.generation_config.pad_token_id = None  # generate tokens without truncation / padding
    
    if args.optimizer == "adam":
        optimizer = optim.Adam(policy.parameters(), lr=args.lr, eps=args.eps)
    elif args.optimizer == "adamw":
        optimizer = optim.AdamW(policy.parameters(), lr=args.lr, eps=args.eps)

    dataset = load_dataset(args.task.query_dataset, split="train")
    train_val_split = dataset.train_test_split(test_size=0.1, seed=args.seed) # use a consistent seed across runs
    dataset, validation_dataset = train_val_split["train"], train_val_split["test"]
    dataset = dataset.with_format("torch", columns=["instruction", "llama_prompt_tokens"])
    dataloader = DataLoader(dataset, batch_size=args.local_batch_size, shuffle=True)
    validation_dataset = validation_dataset.with_format("torch", columns=["instruction", "llama_prompt_tokens"])
    validation_dataloader = DataLoader(validation_dataset, batch_size=args.per_device_eval_batch_size)

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
        temperature=(args.task.temperature + 1e-7),
        top_k=0.0,
        top_p=1.0,
        do_sample=True,
    )
    # use the same `0.01` temperature for validation response generation https://github.com/openai/summarize-from-feedback/blob/700967448d10004279f138666442bf1497d0e705/exps/sample.py#L27
    validation_generation_config = GenerationConfig(
        max_new_tokens=args.task.response_length,
        min_new_tokens=args.task.response_length,
        temperature=(0.01 + 1e-7),
        top_k=0.0,
        top_p=1.0,
        do_sample=True,
    )

    accelerator.print("===training policy===")
    global_step = 0
    start_time = time.time()

    model.train()
    for update in trange(1, args.ppo.num_updates + 1):
        global_step += 1 * args.batch_size
        frac = 1.0 - ((update - 1.0) / args.ppo.num_updates)
        lrnow = frac * args.lr # linear learning rate decay
        optimizer.param_groups[0]["lr"] = lrnow
        data = next(iter_dataloader)
        with torch.no_grad():
            print("sampling evaluation")
            eval_storage, eval_df = evaluate(
                args = args,
                reward_model = reward_model,
                policy = accelerator.unwrap_model(model),
                tokenizer = tokenizer,
                dataloader = validation_dataloader,
                generation_config = validation_generation_config,
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
            queries = data["llama_prompt_tokens"].to(device)
            instructions = data["instruction"]
            context_length = queries.shape[1]
            query_responses = []
            responses = []
            postprocessed_responses = []
            logprobs = []
            ref_logprobs = []
            scores = []
            sequence_lengths = []
            baselines = []
            for i in range(0, queries.shape[0], args.local_rollout_forward_batch_size):
                query = queries[i : i + args.local_rollout_forward_batch_size]
                query_response = generate(
                    accelerator.unwrap_model(model),
                    query,
                    tokenizer,
                    generation_config,
                    n_outputs_per_prompt=args.rloo_k,
                )
                instruction_batch = instructions[i : i + args.local_rollout_forward_batch_size]
                response = query_response[:, context_length:]

                debug_tensor_info(query_response, "query_response")
                output = forward(accelerator.unwrap_model(model), query_response, tokenizer)
                logits = output.logits[:, context_length - 1 : -1]
                logits /= (args.task.temperature + 1e-7)
                all_logprob = F.log_softmax(logits, dim=-1)
                logprob = torch.gather(all_logprob, 2, response.unsqueeze(-1)).squeeze(-1)
                del output, logits, all_logprob
                torch.cuda.empty_cache()

                ref_output = forward(accelerator.unwrap_model(model), query_response, tokenizer, ref=True)
                ref_logits = ref_output.logits[:, context_length - 1 : -1]
                ref_logits /= args.task.temperature + 1e-7
                ref_all_logprob = F.log_softmax(ref_logits, dim=-1)
                ref_logprob = torch.gather(ref_all_logprob, 2, response.unsqueeze(-1)).squeeze(-1)
                del ref_output, ref_logits, ref_all_logprob
                torch.cuda.empty_cache()

                # Response Processing 1. truncate response after the first occurrence of `truncate_token_id`
                postprocessed_response = truncate_response(args, tokenizer, response)

                # Response Processing 2. run reward model on the truncated responses
                repeated_instructions = []
                for inst in instruction_batch:
                    repeated_instructions.extend([inst] * args.rloo_k) # effectively torch.repeat_interleave
                decoded_responses = tokenizer.batch_decode(postprocessed_response, skip_special_tokens=True)
                
                sequence_length = first_true_indices(postprocessed_response == tokenizer.pad_token_id) - 1
                score = get_reward(reward_model = reward_model, 
                                    instruction_str = repeated_instructions,
                                    response_str = decoded_responses,
                                    rm_tokenizer = rm_tokenizer)

                # Calculate baselines
                if args.rloo_k > 1:
                    # The shape of score is [batch_size * rloo_k]
                    per_prompt_scores = score.reshape(-1, args.rloo_k)
                    baseline = (per_prompt_scores.sum(dim = 1, keepdim = True) - per_prompt_scores) / (args.rloo_k - 1)
                    baseline = baseline.reshape(-1)
                else:
                    baseline = torch.zeros_like(score)

                query_responses.append(query_response)
                responses.append(response)
                postprocessed_responses.append(postprocessed_response)
                logprobs.append(logprob)
                ref_logprobs.append(ref_logprob)
                sequence_lengths.append(sequence_length)
                scores.append(score)
                baselines.append(baseline)

            query_responses = torch.cat(query_responses, 0)
            responses = torch.cat(responses, 0)
            postprocessed_responses = torch.cat(postprocessed_responses, 0)
            logprobs = torch.cat(logprobs, 0)
            ref_logprobs = torch.cat(ref_logprobs, 0)
            sequence_lengths = torch.cat(sequence_lengths, 0)
            scores = torch.cat(scores, 0)
            baselines = torch.cat(baselines, 0)
            del (logprob, ref_logprob, score, baseline)
            torch.cuda.empty_cache()

            # Response Processing 3. filter response. Ensure that the sample contains truncate_token_id
            # responses not passing that filter will receive a low (fixed) score
            # only query humans on responses that pass that filter
            contain_pad_token = torch.any(postprocessed_responses == tokenizer.pad_token_id, dim=-1)
            scores = torch.where(contain_pad_token, scores, torch.full_like(scores, args.task.penalty_reward_value))
            accelerator.print(f"{scores=}, {(contain_pad_token.sum() / len(contain_pad_token))=}")

            # 4. compute rewards
            kl = logprobs - ref_logprobs # [batch_size, response_len]
            non_score_reward = -kl_ctl.value * kl
            rewards = non_score_reward.clone()
            actual_start = torch.arange(rewards.size(0), device=rewards.device)
            actual_end = sequence_lengths
            rewards[[actual_start, actual_end]] += scores
            writer.add_scalar("generation/seq_len_mean", sequence_lengths.to(torch.float32).mean().item(), update)
            writer.add_scalar("generation/seq_len_std", sequence_lengths.to(torch.float32).std().item(), update)

            torch.cuda.empty_cache()

        # center = 0.1 * torch.mean(torch.sum(non_score_reward, axis=1) + scores)
        # Do multiple epochs of PPO training, with a fresh random shuffle in each epoch

        stats_shape = (args.ppo.noptepochs)
        metrics = defaultdict(lambda: torch.zeros(stats_shape, device = device))
        
        for ppo_epoch_idx in range(args.ppo.noptepochs):
            local_batch_idxs = np.random.permutation(args.local_batch_size)
            for mini_batch_start in range(0, args.local_batch_size, args.per_device_train_batch_size):
                mini_batch_end = mini_batch_start + args.per_device_train_batch_size
                mini_batch_inds = local_batch_idxs[mini_batch_start:mini_batch_end]
                with accelerator.accumulate(model):
                    # These are all fixed and won't get gradients
                    mb_responses = responses[mini_batch_inds] # [batch_size, response_len]
                    mb_query_responses = query_responses[mini_batch_inds] # [batch_size, seq_len]
                    mb_logprobs = torch.sum(logprobs[mini_batch_inds], axis=1) # [batch_size]
                    mb_ref_logprobs = torch.sum(ref_logprobs[mini_batch_inds], axis=1)
                    mb_reward = scores[mini_batch_inds]
                    mb_baseline = baselines[mini_batch_inds]

                    # compute the logprobs w/ gradient tracking
                    debug_tensor_info(mb_query_responses, "mb_query_responses")
                    output = forward(accelerator.unwrap_model(model), mb_query_responses.clone().detach(), tokenizer)
                    # output.logits has shape [batch_size, seq_len, vocab_size]
                    logits = output.logits[:, context_length-1:-1] # logits of response [batch_size, response_len, vocab_size]
                    logits /= (args.task.temperature + 1e-7)
                    new_all_logprobs = F.log_softmax(logits, dim=-1) # [batch_size, response_len, vocab_size]
                    new_logprobs = torch.sum( # index logprobs over vocab dim by what the model actually generated
                        torch.gather(new_all_logprobs, 2, mb_responses.unsqueeze(-1)).squeeze(-1), axis=1
                    ) # shape [batch_size] (total logprob of the response)

                    if args.train_dips:
                        # the IPS trick loss
                        approx_kl = new_logprobs - mb_ref_logprobs
                        prob_ratio = torch.exp(new_logprobs - mb_logprobs)
                        weighting = (mb_reward - mb_baseline - kl_ctl.value * approx_kl)
                        # loss = torch.mean(-1 * prob_ratio * weighting)
                        policy_loss_term = -1 * (prob_ratio * weighting.detach()).mean()
                        kl_loss_term = -1 * (prob_ratio.detach() * weighting).mean()
                        loss = policy_loss_term + kl_loss_term

                    else:
                        # RLOO loss
                        approx_kl = mb_logprobs - mb_ref_logprobs
                        weighting = (mb_reward - mb_baseline - kl_ctl.value * approx_kl)
                        loss = torch.mean(-1*new_logprobs * weighting)


                    # Grab model grad norms
                    if args.train_dips:
                        accelerator.backward(policy_loss_term, retain_graph = True)
                        policy_term_grad_norms = get_grad_norms(param_subset, device = device)
                        optimizer.zero_grad()

                        accelerator.backward(kl_loss_term, retain_graph = True)
                        kl_term_grad_norms = get_grad_norms(param_subset, device = device)
                        optimizer.zero_grad()

                    accelerator.backward(loss, retain_graph = True) # retain graph to save intermediate grad norms
                    grad_norms = get_grad_norms(param_subset, device = device)
                    #accelerator.clip_grad_norm_(model.parameters(), 1.0)
                    
                    optimizer.step()
                    optimizer.zero_grad()

                    with torch.no_grad():
                        # Do whatever logging we want
                        metrics["loss"][ppo_epoch_idx] += loss.detach()
                        metrics["baseline"][ppo_epoch_idx] += mb_baseline.mean()
                        metrics["grad_norm_mean"][ppo_epoch_idx] += grad_norms.mean()
                        metrics["grad_norm_max"][ppo_epoch_idx] += grad_norms.max()
                        metrics["grad_norm_std"][ppo_epoch_idx] += grad_norms.std()

                        if args.train_dips:
                            metrics["policy_loss_term"][ppo_epoch_idx] += policy_loss_term
                            metrics["kl_loss_term"][ppo_epoch_idx] += kl_loss_term
                            metrics["policy_grad_norm_mean"][ppo_epoch_idx] += policy_term_grad_norms.mean()
                            metrics["policy_grad_norm_max"][ppo_epoch_idx] += policy_term_grad_norms.max()
                            metrics["policy_grad_norm_std"][ppo_epoch_idx] += policy_term_grad_norms.std()
                            metrics["kl_grad_norm_mean"][ppo_epoch_idx] += kl_term_grad_norms.mean()
                            metrics["kl_grad_norm_max"][ppo_epoch_idx] += kl_term_grad_norms.max()
                            metrics["kl_grad_norm_std"][ppo_epoch_idx] += kl_term_grad_norms.std()

                        if args.train_dips:
                            metrics["weighting"][ppo_epoch_idx] += weighting.mean()
                            metrics["prob_ratio"][ppo_epoch_idx] += prob_ratio.mean()
                            metrics["approx_kl"][ppo_epoch_idx] += approx_kl.mean()
                        else:
                            metrics["weighting"][ppo_epoch_idx] += weighting.mean()
                            metrics["new_logprobs"][ppo_epoch_idx] += new_logprobs.mean()
                            metrics["approx_kl"][ppo_epoch_idx] += approx_kl.mean()

                        for key in metrics:
                            metrics[key] /= args.gradient_accumulation_steps
                    
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
                writer.add_scalar(f"train/{stats}", accelerator.gather(metrics[stats]).mean().item(), update)

            if args.reward.use_adaptive_kl:
                kl_ctl.update(mean_kl.item(), args.batch_size)
            del kl, mean_kl, mean_entropy, mean_non_score_reward, scores
            torch.cuda.empty_cache()

    if args.run_eval:
        eval_storage, eval_df = evaluate(
            args,
            reward_model,
            accelerator.unwrap_model(model),
            tokenizer,
            validation_dataloader,
            validation_generation_config,
            sampling=False,
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
