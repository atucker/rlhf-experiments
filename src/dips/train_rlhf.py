from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding
from torch.utils.data import DataLoader
from buffers import split_dicts, fuse_dicts, list_to_batch, to_device, dict_length, reduce_dict, accumulate_dict, slice_dict, gather_dict
import torch
from peft import get_peft_model, PeftModel, LoraConfig
import os
from accelerate import Accelerator
from tqdm import tqdm
import wandb
import argparse

import cProfile


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
    # really, the prompt mask should go here
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
    REFERENCE_ADAPTER_NAME = "reference"
    POLICY_ADAPTER_NAME = "policy"
    REWARD_ADAPTER_NAME = "reward"
    EXPERT_ADAPTER_NAME = "expert"

    def __init__(self, model_name, reference_adapter='', expert_adapter='', dtype=torch.bfloat16):
        super().__init__()
        
        self._model = AutoModelForCausalLM.from_pretrained(model_name).to(dtype)
        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct", padding_side="left")
        self.tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

        peft_config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.1)
        self._model = get_peft_model(self._model, peft_config, self.POLICY_ADAPTER_NAME)

        self.beta = 0.1
        self.temperature = 1

        self.has_reference_adapter = reference_adapter != ""
        if reference_adapter:
            self._model.load_adapter(reference_adapter, adapter_name=self.REFERENCE_ADAPTER_NAME, is_trainable=False, torch_dtype=dtype)
            # Set the initial policy to the reference adapter too
            self._model.load_adapter(reference_adapter, adapter_name=self.POLICY_ADAPTER_NAME, is_trainable=True, torch_dtype=dtype)
            
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
        ans = {
            'input_ids': torch.masked_fill(batch['input_ids'], padding, self.tokenizer.eos_token_id),
            'attention_mask': batch["attention_mask"] * (~padding)
        }
        if 'response_mask' in batch:
            ans['response_mask'] = batch["response_mask"]
        
        return ans

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
        if 'response_mask' in data:
            logprob *= data['response_mask']
        return torch.sum(logprob, axis=1)

    def reward(self, batch):
        with torch.no_grad():
            with self._model.disable_adapter():
                ref_logprobs = self._logprobs(batch)
        self._model.set_adapter(self.EXPERT_ADAPTER_NAME)
        logprobs = self._logprobs(batch)

        return self.beta * (logprobs - ref_logprobs)
        
    
    def save(self, path):
        self._model.save_pretrained(path, is_main_process=True)
        if hasattr(self, '_score_layer'):
            torch.save(self._score_layer.state_dict(), os.path.join(path, "reward", 'score_layer.pt'))


if __name__ == "__main__":
    MAX_GRAD_NORM = 10.0
    MAX_PROMPT_LENGTH = 356
    MAX_RESPONSE_LENGTH = 156

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--algorithm', 
        choices=['dips', 'rloo'],  # Only these values are allowed
        default='dips',
        help='algorithm to use'
    )
    parser.add_argument('--reference', default="")
    parser.add_argument('--expert', default="expert/expert")
    parser.add_argument('--beta', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=2e-5)
    # defaults are for single H100 in float32
    parser.add_argument('--k', type=int, default=2)
    parser.add_argument('--train_batch_size', type=int, default=1)
    parser.add_argument('--score_batch_size', type=int, default=16)
    parser.add_argument('--generate_batch_size', type=int, default=32)
    parser.add_argument('--update_batch_size', type=int, default=128)
    parser.add_argument('--bfloat16', action='store_true', help='Use bfloat16')
    parser.add_argument('--instruct_base', action='store_true', help='Use the instruct model')
    args = parser.parse_args()

    # the generations need to cleanly break into training batches
    assert (args.generate_batch_size * args.k) % args.train_batch_size == 0
    # every group of k generations needs to fit in a score batch
    assert args.score_batch_size % args.k == 0
    # the updates need to consist of entire generations
    assert args.update_batch_size % (args.generate_batch_size * args.k) == 0

    dtype = torch.bfloat16 if args.bfloat16 else torch.float32
    DO_ARMO = args.expert.upper() == "ARMO"
    print(args.expert.upper())
    if DO_ARMO:
        armo = AutoModelForSequenceClassification.from_pretrained(
            "RLHFlow/ArmoRM-Llama3-8B-v0.1", trust_remote_code=True,
        ).to(dtype)
    armo_tokenizer = AutoTokenizer.from_pretrained("RLHFlow/ArmoRM-Llama3-8B-v0.1", padding_side="left")
    armo_tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

    wandb.init(
        project="8b-demo",
        name=f"train_{args.algorithm}"
    )

    model_name = "meta-llama/Llama-3.1-8B-Instruct" if args.instruct_base else "meta-llama/Llama-3.1-8B"
    
    dataset = load_dataset("openbmb/UltraFeedback")
    #model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct").to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
    
    prompt_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest")
    response_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest")
    
    dataset = dataset.filter(length_filter)
    dataloader = DataLoader(
        dataset["train"].map(process_prompts, batched=False, remove_columns=dataset["train"].column_names),
        batch_size=args.generate_batch_size, collate_fn=collate_fn, shuffle=True
    )
    
    model = Model(
        model_name, reference_adapter=args.reference, expert_adapter=args.expert if not DO_ARMO else "",
        dtype=dtype
    )
    
    accelerator = Accelerator()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    warmup = len(dataloader) / 10
    scheduler = LambdaLR(optimizer, lambda step: min(1, step / warmup))
    if not DO_ARMO:
        model, optimizer, scheduler, dataloader = accelerator.prepare(model, optimizer, scheduler, dataloader)
    else:
        model, optimizer, scheduler, dataloader, armo = accelerator.prepare(model, optimizer, scheduler, dataloader, armo)
    

    i = 0
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
                num_return_sequences=args.k,
                temperature=1,
                max_new_tokens=MAX_RESPONSE_LENGTH,
                do_sample=True,
                use_cache=True,          # Enable KV caching
                num_beams=1,             # Ensure no beam search
            )

            attention_mask = torch.ones(response.shape).to(accelerator.device)
            attention_mask[:, :inpts['attention_mask'].shape[1]] = torch.repeat_interleave(inpts['attention_mask'], repeats=args.k, axis=0)

            prompt_mask  = torch.zeros(response.shape).to(accelerator.device)
            prompt_mask[:, :inpts['input_ids'].shape[1]] = 1
            
            outputs = model._mask_padding({
                'input_ids': response,
                'attention_mask': attention_mask,
                'prompt_mask': prompt_mask
            }, [128001])
            # , model.tokenizer.eos_token_id

            start = 0
            score_dict, log_info = {}, {}
            while start + args.score_batch_size <= dict_length(outputs):
                batch, info = {}, {}
                minibatch = slice_dict(outputs, start, start + args.score_batch_size)
                
                # Compute our no-grad scores and ref logprobs
                # pretty bad implementation, should be abstracted
                if model.has_reference_adapter:
                    model._model.set_adapter(model.REFERENCE_ADAPTER_NAME)
                    batch['ref_logprobs'] = model._logprobs(minibatch)
                else:
                    with model._model.disable_adapter():
                        batch['ref_logprobs'] = model._logprobs(minibatch)
    
                if not DO_ARMO:
                    model._model.set_adapter(model.EXPERT_ADAPTER_NAME)
                    batch['expert_logprobs'] = model._logprobs(minibatch)
                    batch['scores'] = model.beta * (batch['expert_logprobs'] - batch['ref_logprobs'])
                else:
                    reward_info = armo(
                        input_ids=minibatch['input_ids'], 
                        attention_mask=minibatch['attention_mask'],
                        return_dict=True
                    )
                    batch['scores']  = reward_info.score
                    info = {**info, **reward_info}

                model._model.set_adapter(model.POLICY_ADAPTER_NAME)
                batch['sample_logprobs'] = model._logprobs(minibatch)

                batch["kls"] = batch['sample_logprobs'] - batch['ref_logprobs']
                rewards = batch["rewards"] = batch['scores'] - model.beta * batch["kls"]

                # Setup the baseline and center the rewards
                # we do this in the non-differentiable setting because you can't actually
                # compute this in the differentiable setting if train_batch_size < k
                assert rewards.shape[0] % args.k == 0
                baseline_sum = rewards.reshape(int(rewards.shape[0] / args.k), args.k).sum(axis=1)
                batch["baselines"] = (
                    baseline_sum.repeat_interleave(repeats=args.k, axis=0) - rewards
                ) / (args.k - 1)

                start += args.score_batch_size
                # probably we should keep these as a list and iterate through them rather than concatenating
                score_dict = accumulate_dict(score_dict, batch)
                log_info = accumulate_dict(score_dict, info)
                
            outputs = {**outputs, **score_dict}
    
        if accelerator.is_main_process:
            print("Training...")

        model.eval()
        start = 0
        while start + args.train_batch_size <= dict_length(outputs):
            minibatch = slice_dict(outputs, start, start + args.train_batch_size)
            info = slice_dict(log_info, start, start + args.train_batch_size)

            # Compute the reward
            model._model.set_adapter(model.POLICY_ADAPTER_NAME)
            policy_logprobs = model._logprobs(minibatch)
            if args.algorithm == "dips":
                # recompute the reward for a differentiable KL
                kls = policy_logprobs - minibatch["ref_logprobs"]
                rewards = minibatch["scores"] - model.beta * kls
            else:
                kls = batch["kls"]
                rewards = minibatch["rewards"]
            advantages = rewards - minibatch["baselines"]
        
            if args.algorithm == "dips":
                # We should tweak this to cut out the prompts, so that it doesn't just
                # get reward from memorizing prompts
                prob_ratio = torch.exp(policy_logprobs - minibatch["sample_logprobs"])
                loss = -1 * prob_ratio * advantages
                info["prob_ratio"] = prob_ratio.detach()
            elif args.algorithm == "rloo":
                loss = -1 * policy_logprobs * advantages
            else:
                raise NotImplementedError()
    
            accelerator.backward(torch.mean(loss))
    
            assert loss.shape[0] == args.train_batch_size
            n += loss.shape[0]
            start += args.train_batch_size
            
            log_data = accumulate_dict(log_data, {
                'loss': loss.detach(),
                'kl': kls.detach(),
                'scores': minibatch["scores"].detach(),
                'total_reward': rewards.detach(),
                'baselines': minibatch["baselines"].detach(),
                'advantages': advantages.detach(),
                **info
            })
    
            if n % args.update_batch_size == 0:
                log_data = reduce_dict(gather_dict(accelerator, log_data))
                if accelerator.is_main_process:
                    wandb.log(log_data)
                accelerator.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                log_data = {}