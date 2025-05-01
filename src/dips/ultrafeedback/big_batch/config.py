from dataclasses import dataclass, field
from typing import Optional, List, Literal
import tyro.conf

@dataclass
class AdaptiveKLParams:
    target: float = 6.0
    horizon: int = 10000  # in episodes

@dataclass
class RewardHParams:
    use_adaptive_kl: bool = False
    adaptive_kl: Optional[AdaptiveKLParams] = field(default_factory=AdaptiveKLParams)
    kl_coef: float = 5e-3

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
    query_length: int = 256 # Filter out queries longer than this
    query_dataset: str = "openbmb/UltraFeedback"
    chat_template_buffer_length: int = 64

    # Response params
    response_length: int = 1024

    # Truncate response after the first occurrence of this token at or after index after when sampling.
    truncate_token: Literal["eos"] = "eos"
    truncate_token_id: Optional[int] = None
    penalty_reward_value: int = -4

    # LM params
    temperature: float = 0.75

    # Reward scaling
    reward_coef: float = 4.0

@dataclass
class Args:
    train_dips: bool = False # whether to train via DIPS or RLOO
    factor_loss: bool = False
    debug_tensor_info: bool = False
    loss_full_precision: bool = False
    unembed_full_precision: bool = False
    use_chat_template: bool = True
    calculate_kl_on_truncated_responses: bool = True # MUST BE TRUE for big-batch RLHF.
    clip_grad_norm: Optional[float] = None
    force_clear_grad_optim: bool = True # an optimization to reduce GPU memory usage. May mess with gradient clipping.
    kl_grad_patch: bool = False # use RLOO with the KL gradient term patched in. Should be theoretically equivalent to DIPS.
    swap_eos_token: bool = False # necessary if training the base model
    # common args
    exp_name: str = "llama_3_8b_ultrafeedback"
    """the name of this experiment"""
    seed: int = 55134
    """seed of the experiment"""
    track: bool = False
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
    print_sample_output_freq: int = 200
    """How often to print sample output"""
    run_eval: bool = True
    """Whether to run evaluation"""
    max_eval_size: int = 100
    """Maximum number of samples to generate once per print_sample_output_freq (set < 100 for fast evaluation)"""

    # optimizer args
    eps: float = 1e-5
    """the epsilon value for the optimizer - an extremely small value to prevent division by zero"""
    lr: float = 1e-4
    """the learning rate"""
    optimizer: Literal["adam", "adamw"] = "adamw"
    """Which optimizer to use"""
    scheduler: str = "linear"
    """Which scheduler to use"""
    warm_up_steps: int = 50
    """Number of warm up steps for the scheduler"""

    # default args
    batch_size: int = -1

    gradient_accumulation_steps: int = 64
    """The number of gradient accumulation steps"""

    # ------ Batch Size in Memory / GPU: per_device_train_batch_size --------
    rloo_k: int = 2 # number of samples to use for RLOO's baseline calculation
    
    per_device_train_batch_size: int = 2
    """The micro batch size per GPU (HF's `per_device_train_batch_size`)"""
    per_device_eval_batch_size: int = 4
    """per rank eval batch size"""
    per_device_rollout_batch_size: int = 256
    """per rank no grad forward pass in the rollout phase. Note that this is multiplied by rloo_k - we have 8 novel prompts and generate 4 responses for each."""
    local_rollout_forward_batch_size: int = 4

    total_episodes: int = int(6416) # Informs the number of ppo updates to do
    """The total number of episodes in the dataset"""

    # optional args filled while running
    world_size: Optional[int] = 1
    """The number of processes (GPUs) to use"""

    # other args
    base_model: str = "meta-llama/Llama-3.1-8B-Instruct"
    """the name of the pretrained model to use"""
    offload: bool = False
    """Whether to offload ref policy and reward model to CPU"""
    reward_model_path: str = "RLHFlow/ArmoRM-Llama3-8B-v0.1"
    """the name of the pretrained model to use"""
    sft_model_path: str = "meta-llama/Llama-3.1-8B-Instruct"
    """the name of the pretrained model to use"""
    chat_template_tokenizer: Optional[str] = None
    """the name of the tokenizer to use for apply_chat_template"""
    dropout_layer_keys: List[str] = field(
        default_factory=lambda: ["attn_pdrop", "embd_pdrop", "resid_pdrop", "summary_first_dropout"]
    )
    """Which layers to apply dropout to"""
    output_dir: str = "models/llama_3_8b_armoRM_ultrafeedback"
    """Where to save the model"""
    lora_rank: int = 256
    """the rank of the lora matrix"""
    lora_alpha: int = 256
    """weight of lora"""
    lora_dropout: float = 0.0
    """dropout for lora"""
    task: TaskHParams = field(default_factory=TaskHParams)
    reward: RewardHParams = field(default_factory=RewardHParams)
    ppo: PpoHParams = field(default_factory=PpoHParams)