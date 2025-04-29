import pandas as pd
import torch
from rich.console import Console
from rich.table import Table
from typing import Tuple, Dict

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

def parse_reward_breakdown_attributes(reward_breakdown: torch.Tensor, reward_breakdown_coeffs: torch.Tensor) -> Tuple[Dict[str, float], Dict[str, float]]:
    attributes = ['helpsteer-helpfulness','helpsteer-correctness','helpsteer-coherence',
   'helpsteer-complexity','helpsteer-verbosity','ultrafeedback-overall_score',
   'ultrafeedback-instruction_following', 'ultrafeedback-truthfulness',
   'ultrafeedback-honesty','ultrafeedback-helpfulness','beavertails-is_safe',
   'prometheus-score','argilla-overall_quality','argilla-judge_lm','code-complexity',
   'code-style','code-explanation','code-instruction-following','code-readability']
    reward_breakdown_dict = {}
    reward_breakdown_coeffs_dict = {}
    for index, elem in enumerate(attributes):
        reward_breakdown_dict[elem] = reward_breakdown[:, index].mean().item()
        reward_breakdown_coeffs_dict[elem] = reward_breakdown_coeffs[:, index].mean().item()
    return reward_breakdown_dict, reward_breakdown_coeffs_dict