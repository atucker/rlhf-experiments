from dataclasses import dataclass, field
from typing import List, Tuple
import torch
import pandas as pd
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, GenerationConfig
from accelerate import Accelerator
from dips.ultrafeedback.big_batch.config import Args
from dips.ultrafeedback.big_batch.utils import maybe_use_chat_template, truncate_response
from dips.ultrafeedback.big_batch.model import generate
from dips.ultrafeedback.big_batch.reward import get_reward
import torch.nn as nn
from accelerate.utils import gather_object

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
             dataloader: DataLoader, generation_config: GenerationConfig, sampling=True,
             max_eval_size = float('inf'),
             accelerator: Accelerator = None) -> Tuple[EvalStorage, pd.DataFrame]:
    """
    Completes an episode rollout for the policy model and returns:

    - reference response and reference response performance
    - policy-generated response and policy-generated response performance
    - kl divergence between the policy and reference model
    """
    eval_storage = EvalStorage()
    counter = 0
    with torch.no_grad():
        for data in dataloader:
            # 1. Extract queries and reference responses from the dataset
            instruction = data["instruction"]
            queries = maybe_use_chat_template(instruction, 
                                              use_chat_template = args.use_chat_template, 
                                              )
            context_length = queries.shape[1]
            if args.use_chat_template:
                assert context_length == args.task.query_length + args.task.chat_template_buffer_length, f"Context length {context_length} does not match query length {args.task.query_length + args.task.chat_template_buffer_length}"
            else:
                assert context_length == args.task.query_length, f"Context length {context_length} does not match query length {args.task.query_length}"

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
            truncated_query_responses = torch.cat([queries, postprocessed_responses], dim = 1)
            scores, _, _ = get_reward(reward_model = reward_model, 
                               input_ids = truncated_query_responses)
            eval_storage.query_token.extend(queries)
            eval_storage.postprocessed_response_token.extend(postprocessed_responses)
            eval_storage.score.append(scores)

            if sampling:
                break

            counter += 1
            if counter >= max_eval_size:
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