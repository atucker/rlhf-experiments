from dataclasses import asdict, dataclass, field
from typing import List, Literal, Optional, Tuple, Union
import tyro

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
    whiten_rewards: bool = True


@dataclass
class TaskHParams:
    # Query params
    query_length: int = 512
    query_dataset: str = "cleanrl/summarize_from_feedback_tldr_3_filtered_oai_preprocessing_1704563162" # pythia 2.9

    # Response params
    response_length: int = 53

    # Truncate response after the first occurrence of this token at or after index after when sampling.
    truncate_token: Literal["eos"] = "eos"
    truncate_token_id: Optional[int] = None
    penalty_reward_value: int = -1

    # LM params
    temperature: float = 0.7


@dataclass
class Args:
    train_dips: bool = False # whether to train via DIPS or RLOO
    disable_wandb: bool = False
    factor_loss: bool = False
    loss_full_precision: bool = True
    unembed_full_precision: bool = True
    # common args
    exp_name: str = "pythia"
    """the name of this experiment"""
    seed: int = 55134
    """seed of the experiment"""
    track: bool = True
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "tldr_summarize_pythia_will"
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
    lr: float = 3e-6
    """the learning rate"""
    optimizer: Literal["adam", "adamw"] = "adamw"
    """Which optimizer to use"""
    scheduler: str = "cosine"
    """Which scheduler to use"""
    warm_up_steps: int = 0
    """Number of warm up steps for the scheduler"""

    gradient_accumulation_steps: int = 8
    """The number of gradient accumulation steps"""

    # ------ Batch Size in Memory / GPU: per_device_train_batch_size --------
    rloo_k: int = 4 # number of samples to use for RLOO
    
    per_device_train_batch_size: int = 8
    """The micro batch size per GPU (HF's `per_device_train_batch_size`)"""
    per_device_eval_batch_size: int = 8
    """per rank eval batch size"""
    local_rollout_forward_batch_size: int = 8
    """per rank no grad forward pass in the rollout phase"""

    total_episodes: int = int(1e4) # Informs the number of ppo updates to do
    """The total number of episodes in the dataset"""

    # optional args filled while running
    world_size: Optional[int] = 2
    """The number of processes (GPUs) to use"""

    # other args
    base_model: str = "models/sft_tldr_pythia_1_4b"
    """the name of the pretrained model to use"""
    offload: bool = False
    """Whether to offload ref policy and reward model to CPU"""
    reward_model_path: str = "models/rm_sft_tldr_pythia_1_4b"
    """the name of the pretrained model to use"""
    sft_model_path: str = "models/sft_tldr_pythia_1_4b"
    """the name of the pretrained model to use"""
    dropout_layer_keys: List[str] = field(
        default_factory=lambda: ["attn_pdrop", "embd_pdrop", "resid_pdrop", "summary_first_dropout"]
    )
    """Which layers to apply dropout to"""
    output_dir: str = "models/tldr_pythia_1_4b"
    """Where to save the model"""
    lora_rank: int = 1024
    """the rank of the lora matrix"""
    lora_alpha: int = 2048
    """weight of lora"""
    lora_dropout: float = 0.0
    """dropout for lora"""
    task: TaskHParams = field(default_factory=TaskHParams)
    reward: RewardHParams = field(default_factory=RewardHParams)
    ppo: PpoHParams = field(default_factory=PpoHParams)

def validate_args(args: Args):
    if args.factor_loss:
        assert args.train_dips, "Loss factoring only supports DIPS. Are you using --train_dips?"
    assert args.rloo_k >= 1