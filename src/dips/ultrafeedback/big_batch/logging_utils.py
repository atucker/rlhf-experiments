import pandas as pd
import torch
from rich.console import Console
from rich.table import Table
from typing import Tuple, Dict

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

class GradNormLogger:
    def __init__(self, device = "cpu"):
        self.tracked_params = None
        self.device = device

    def setup(self, tracked_params: list):
        self.tracked_params = tracked_params

    def get_grad_norms(self, loss) -> torch.Tensor:
        if self.tracked_params is None:
            raise ValueError("GradNormLogger has not been setup with tracked parameters")
        grad = torch.autograd.grad(
            outputs = loss,
            inputs = self.tracked_params,
            create_graph = False,
            retain_graph = True,
        )
        grad_norms = torch.tensor([g.norm() for g in grad])
        return grad_norms