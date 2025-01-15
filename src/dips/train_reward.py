from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorWithPadding
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from buffers import split_dicts, fuse_dicts, list_to_batch, to_device, dict_length, reduce_dict, accumulate_dict, gather_dict
import torch
from peft import get_peft_model, PeftModel, LoraConfig
import os
from accelerate import Accelerator
import wandb
from tqdm import tqdm



def prompt_chosen_rejected(sample):
    responses = []
    for _ in sample['completions']:
        responses.append({"score": _['overall_score'], "response": _['response']})
    responses = sorted(responses, key=lambda x: x["score"])

    if len(responses) < 2:
        return None

    rejected = responses[0]["response"]
    chosen = responses[-1]["response"]

    prompt = f"User: {sample['instruction']}\n Assistant: "
    return {
        "prompt": prompt,
        "chosen": prompt + chosen,
        "rejected": prompt + rejected
    }


def length_filter(sample):
    parsed = prompt_chosen_rejected(sample)

    if parsed is None:
        return False

    return (
        len(tokenizer(parsed["prompt"]).input_ids) <= MAX_PROMPT_LENGTH  and
        len(tokenizer(parsed["chosen"]).input_ids) <= MAX_RESPONSE_LENGTH + MAX_PROMPT_LENGTH and
        len(tokenizer(parsed["rejected"]).input_ids) <= MAX_RESPONSE_LENGTH + MAX_PROMPT_LENGTH
    )


def tokenize_save_text(text):
    data = tokenizer(text)
    return {
        "input_ids": data.input_ids,
        "attention_mask": data.attention_mask,
        "text": text
    }


def process_prompts(sample):
    parsed = prompt_chosen_rejected(sample)
    ans = fuse_dicts(
        prompt=tokenize_save_text(parsed["prompt"]),
        chosen=tokenize_save_text(parsed["chosen"]),
        rejected=tokenize_save_text(parsed["rejected"]),
    )
    return ans


def collate_skip_text(collate, batch):
    data = dict((key, value) for key, value in batch.items() if key != "text")
    data = collate(data)
    ans = {"text": batch["text"]}
    for key in data.keys():
        ans[key] = data[key]
    return ans


def collate_fn(batch):
    batch = list_to_batch(batch, return_list=True)
    batch = split_dicts(batch, ['prompt', 'chosen', 'rejected'])
    return fuse_dicts(
        prompt=collate_skip_text(prompt_collator, batch['prompt']),
        chosen=collate_skip_text(response_collator, batch['chosen']),
        rejected=collate_skip_text(response_collator, batch['rejected'])
    )


class Model(torch.nn.Module):
    POLICY_ADAPTER_NAME = "policy"
    REWARD_ADAPTER_NAME = "reward"
    EXPERT_ADAPTER_NAME = "expert"

    def __init__(self, model_name, accelerator, expert_adapter='', dtype=torch.bfloat16):
        super().__init__()
        self.accelerator = accelerator
        
        self._model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype).to(accelerator.device)
        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct", padding_side="left")
        self.tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

        peft_config = LoraConfig(r=8, lora_alpha=64, lora_dropout=0.1)
        self._model = get_peft_model(self._model, peft_config, self.POLICY_ADAPTER_NAME)

        self.beta = 0.1
        self.temperature = 1

        if expert_adapter:
            #self._model.load_adapter(reward_adapter, adapter_name=self.REWARD_ADAPTER_NAME, is_trainable=(reward_adapter == ''), torch_dtype=torch.bfloat16)
            #state_dict = torch.load(os.path.join(classifier_adapter, 'score_layer.pt'),  map_location=accelerator.device)
            #self._score_layer = torch.nn.Linear(self._model.config.hidden_size, 1, dtype=dtype)
            #self._score_layer.load_state_dict(state_dict)
            self._model.load_adapter(expert_adapter, adapter_name=self.EXPERT_ADAPTER_NAME, is_trainable=False, torch_dtype=dtype)
        else:
            self._model.add_adapter(self.EXPERT_ADAPTER_NAME, peft_config)
            #self._score_layer = torch.nn.Linear(self._model.config.hidden_size, 1, dtype=dtype).to(DEVICE)


    def _mask_padding(self, batch):
        padding = batch['input_ids'] == self.tokenizer.pad_token_id
        return {
            'input_ids': torch.masked_fill(batch['input_ids'], padding, 0),
            'attention_mask': batch["attention_mask"] * (~padding)
        }

    def _logprobs(self, batch):
        data = self._mask_padding(batch)
        logits = self._model.forward(**data).logits
        # Pad output sequence to response_length (necessary to avoid shape mismatch across devices)
        # pad = torch.zeros(logits.shape[0], args.task.response_length-logits.shape[1], dtype = logits.dtype).to(device)
        logits /= self.temperature
        all_logprob = torch.nn.functional.log_softmax(logits, dim=-1)
        logprob = torch.gather(all_logprob, 2, data['input_ids'].unsqueeze(-1)).squeeze(-1)
        logprob *= data['attention_mask']
        assert logprob.shape == batch['input_ids'].shape
        return torch.sum(logprob, axis=1)

    def reward(self, batch):
        #self._model.set_adapter(self.REWARD_ADAPTER_NAME)
        #hidden = self._model(
        #    **self._mask_padding(batch),
        #    output_hidden_states=True
        #).hidden_states[-1]
        #return torch.sum(self._score_layer(hidden).squeeze(), axis=1
        with torch.no_grad():
            with self._model.disable_adapter():
                ref_logprobs = self._logprobs(batch)
        self._model.set_adapter(self.EXPERT_ADAPTER_NAME)
        logprobs = self._logprobs(batch)

        return self.beta * (logprobs - ref_logprobs)
        
    
    def save(self, path):
        self._model.save_pretrained(path, is_main_process=self.accelerator.is_main_process)
        if hasattr(self, '_score_layer'):
            torch.save(self._score_layer.state_dict(), os.path.join(path, "reward", 'score_layer.pt'))


if __name__ == "__main__":
    MAX_PROMPT_LENGTH = 256
    MAX_RESPONSE_LENGTH = 1024
    GENERATE_BATCH_SIZE = 8
    GRAD_BATCH_SIZE = 4
    UPDATE_BATCH_SIZE = 64
    SAVE_FREQ = 5000
    MAX_GRAD_NORM = 10.0
    #N_TRAIN = 60000
    wandb.init(
        project="8b-demo",
        name="train_expert"
    )
    
    dataset = load_dataset("openbmb/UltraFeedback")
    #model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct").to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct", padding_side="left")
    tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
    
    prompt_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest")
    response_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest")
    
    dataset = dataset.filter(length_filter)
    dataloader = DataLoader(
        dataset["train"].map(process_prompts, batched=False, remove_columns=dataset["train"].column_names),
        batch_size=GRAD_BATCH_SIZE, collate_fn=collate_fn, shuffle=True
    )

    accelerator = Accelerator()
    model = Model("meta-llama/Llama-3.1-8B-Instruct", accelerator)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-7)
    warmup = len(dataloader) / 10
    scheduler = LambdaLR(optimizer, lambda step: min(1, step / warmup))
    model, optimizer, scheduler, dataloader = accelerator.prepare(model, optimizer, scheduler, dataloader)

    n = 0
    save_n = 0

    accelerator.unwrap_model(model).train()
    log_data = {}
    for batch in tqdm(dataloader):
        #batch = to_device(batch, DEVICE)
        chosen, rejected = split_dicts(batch, ['chosen', 'rejected'], unpack=True)
        chosen_reward = accelerator.unwrap_model(model).reward(chosen)
        rejected_reward = accelerator.unwrap_model(model).reward(rejected)
        loss = -1 * torch.nn.functional.logsigmoid(chosen_reward - rejected_reward)
        assert len(loss.shape) == 1
        assert UPDATE_BATCH_SIZE % loss.shape[0] == 0
        n += loss.shape[0] * accelerator.num_processes
    
        accelerator.backward(torch.mean(loss))
    
        log_data = accumulate_dict(log_data, {
            'loss': loss.detach(),
            'chosen_reward': chosen_reward.detach(),
            'rejected_reward': rejected_reward.detach(),
            'accuracy': (chosen_reward > rejected_reward).detach()
        })
    
        if n % UPDATE_BATCH_SIZE == 0:
            log_data = reduce_dict(gather_dict(accelerator, log_data))
            if accelerator.is_main_process:
                wandb.log(log_data)

            accelerator.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            log_data = {}

        if n > save_n:
            save_n += SAVE_FREQ
            accelerator.unwrap_model(model).save(f'test/checkpoints/{n}')
    accelerator.unwrap_model(model).save(f'expert')