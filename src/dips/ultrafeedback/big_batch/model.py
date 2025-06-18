import torch
import torch.optim as optim
from transformers import AutoModelForCausalLM, AutoConfig
from accelerate import Accelerator
from dips.ultrafeedback.big_batch.config import Args
from peft import get_peft_model, LoraConfig
from dips.ultrafeedback.big_batch.logging_utils import GradNormLogger
from transformers import get_scheduler
from peft import PeftModel

class PrecisionModel(AutoModelForCausalLM):
    def forward(self, *args, **kwargs):
        before_unembed = super().forward(*args, **kwargs, output_hidden_states = True).hidden_states[-1]
        with torch.amp.autocast(device_type = "cuda", enabled = False):
            before_unembed = before_unembed.to(torch.float32)
            logits = self.lm_head(before_unembed)
        return logits
    
def initialize_policy_with_optimizer(args: Args,
                                    model_config: AutoConfig,
                                    grad_norm_logger: GradNormLogger,
                                    lora_dir: str = None,
                                    load_from_checkpoint: bool = False):
    """
    Returns: policy, optimizer, scheduler.
    """

    if args.unembed_full_precision:
        policy = PrecisionModel.from_pretrained(args.sft_model_path,
                                                config=model_config,
                                                trust_remote_code=True) 
    else:
        policy = AutoModelForCausalLM.from_pretrained(args.sft_model_path, 
                                                    config=model_config, 
                                                    trust_remote_code=True,
                                                    torch_dtype="auto") 

    # Freeze the policy model base weights
    for param in policy.parameters():
        param.requires_grad = False

    peft_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
    )
    if load_from_checkpoint:
        policy = PeftModel.from_pretrained(policy, 
                                           lora_dir, 
                                           config = peft_config,
                                           is_trainable = True)
    else:
        policy = get_peft_model(policy, peft_config=peft_config)

    param_subset = [param for param in policy.parameters() if param.requires_grad]
    grad_norm_logger.setup(param_subset)

    policy.generation_config.eos_token_id = None  # disable `pad_token_id` and `eos_token_id` because we just want to
    policy.generation_config.pad_token_id = None  # generate tokens without truncation / padding
    
    if args.optimizer == "adam":
        optimizer = optim.Adam(param_subset, lr=args.lr, eps=args.eps)
    elif args.optimizer == "adamw":
        optimizer = optim.AdamW(param_subset, lr=args.lr, eps=args.eps)

    # print("Params being optimized:", [name for name, param in policy.named_parameters() if param.requires_grad])

    scheduler = get_scheduler(
        args.scheduler,
        optimizer = optimizer,
        num_warmup_steps = args.warm_up_steps,
        num_training_steps = args.ppo.num_updates,
    )

    return policy, optimizer, scheduler