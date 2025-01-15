from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorWithPadding
from torch.utils.data import DataLoader
from buffers import split_dicts, fuse_dicts, list_to_batch, to_device, dict_length, reduce_dict, accumulate_dict, slice_dict, gather_dict
import torch
from peft import get_peft_model, PeftModel, LoraConfig
import os
from accelerate import Accelerator
from tqdm import tqdm
import wandb
import argparse


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

    def __init__(self, model_name, expert_adapter='', dtype=torch.bfloat16):
        super().__init__()
        
        self._model = AutoModelForCausalLM.from_pretrained(model_name).to(dtype)
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

    def _mask_padding(self, batch, mask_tokens=[]):
        mask_tokens = mask_tokens + [self.tokenizer.pad_token_id]
        assert len(set(mask_tokens)) == len(mask_tokens)
        
        padding = batch['input_ids'] != batch['input_ids']
        for token in mask_tokens:
            padding |= batch['input_ids'] == token
        return {
            'input_ids': torch.masked_fill(batch['input_ids'], padding, self.tokenizer.bos_token_id),
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

        return (logprobs - ref_logprobs) / self.beta
        
    
    def save(self, path):
        self._model.save_pretrained(path, is_main_process=True)
        if hasattr(self, '_score_layer'):
            torch.save(self._score_layer.state_dict(), os.path.join(path, "reward", 'score_layer.pt'))


if __name__ == "__main__":
    MAX_PROMPT_LENGTH = 256
    MAX_RESPONSE_LENGTH = 1024
    GENERATE_BATCH_SIZE = 64
    TRAIN_BATCH_SIZE = 2
    UPDATE_BATCH_SIZE = 128
    k = 2

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--algorithm', 
        choices=['dips', 'rloo'],  # Only these values are allowed
        default='dips',
        help='algorithm to use'
    )
    parser.add_argument('--expert', default="test/checkpoints/15008/expert")
    args = parser.parse_args()

    wandb.init(
        project="8b-demo",
        name=f"train_{args.algorithm}"
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
        batch_size=GENERATE_BATCH_SIZE, collate_fn=collate_fn, shuffle=True
    )
    
    model = Model("meta-llama/Llama-3.1-8B-Instruct", expert_adapter=args.expert)
    
    accelerator = Accelerator()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    

    n = 0
    log_data = {}
    for batch in tqdm(dataloader):
        #batch = to_device(batch, DEVICE)
        prompt, = split_dicts(batch, ["prompt"], unpack=True)
    
        model.eval()
        with torch.no_grad():
            if accelerator.is_main_process:
                print("Sampling...")
            # Generate the samples and build an attention mask
            inpts = model._mask_padding(prompt)
            model._model.set_adapter(model.POLICY_ADAPTER_NAME)
            response = model._model.generate(
                **inpts,
                #top_k=50,
                num_return_sequences=k,
                temperature=1,
                max_new_tokens=512,
                do_sample=True,
                use_cache=True,          # Enable KV caching
                num_beams=1,            # Ensure no beam search
            )
            
            attention_mask = torch.ones(response.shape).to(accelerator.device)
            attention_mask[:, :inpts['attention_mask'].shape[1]] = torch.repeat_interleave(inpts['attention_mask'], repeats=k, axis=0)
            
            outputs = model._mask_padding({
                'input_ids': response,
                'attention_mask': attention_mask
            }, [128001, model.tokenizer.eos_token_id])
    
        # the generations need to cleanly break into training batches
        assert (GENERATE_BATCH_SIZE * k) % TRAIN_BATCH_SIZE == 0
        # the training batches need to fit all k
        assert TRAIN_BATCH_SIZE % k == 0
        # the updates need to consist of entire generations
        assert UPDATE_BATCH_SIZE % (GENERATE_BATCH_SIZE * k) == 0
    
        if accelerator.is_main_process:
            print("Training...")
        model.eval()
        start = 0
        while start + TRAIN_BATCH_SIZE <= dict_length(outputs):
            minibatch = slice_dict(outputs, start, start + TRAIN_BATCH_SIZE)
            with torch.no_grad():
                # Compute our no-grad scores and ref logprobs
                with model._model.disable_adapter():
                    ref_logprobs = model._logprobs(minibatch)
                model._model.set_adapter(model.EXPERT_ADAPTER_NAME)
                expert_logprobs = model._logprobs(minibatch)
                scores = model.beta * (expert_logprobs - ref_logprobs)
                model._model.set_adapter(model.POLICY_ADAPTER_NAME)
                sample_logprobs = model._logprobs(minibatch)
            
            # Compute the reward
            model._model.set_adapter(model.POLICY_ADAPTER_NAME)
            policy_logprobs = model._logprobs(minibatch)
            with torch.set_grad_enabled(args.algorithm=="dips"):
                kls = 0.1 * (policy_logprobs - ref_logprobs)
                rewards = scores - kls
        
            # Setup the baseline and center the rewards
            with torch.no_grad():
                baselines = rewards.reshape(int(rewards.shape[0] / k), k).sum(axis=1) / (k-1)
                baselines = baselines.repeat_interleave(repeats=k, axis=0)
            advantages = (k/(k-1))*rewards - baselines
        
            info = {}
            if args.algorithm == "dips":
                prob_ratio = torch.exp(policy_logprobs - sample_logprobs)
                loss = -1 * prob_ratio * advantages
                info = {"prob_ratio": prob_ratio.detach()}
            elif args.algorithm == "rloo":
                loss = -1 * policy_logprobs * advantages
            else:
                raise NotImplementedError()
    
            accelerator.backward(torch.mean(loss))
    
            assert loss.shape[0] == TRAIN_BATCH_SIZE
            n += loss.shape[0]
            start += TRAIN_BATCH_SIZE
            
            log_data = accumulate_dict(log_data, {
                'loss': loss.detach(),
                'kl': kls.detach(),
                'scores': scores.detach(),
                'total_reward': rewards.detach(),
                **info
            })
    
            if n % UPDATE_BATCH_SIZE == 0:
                log_data = reduce_dict(gather_dict(accelerator, log_data))
                if accelerator.is_main_process:
                    wandb.log(log_data)
                
                optimizer.step()
                optimizer.zero_grad()
                log_data = {}