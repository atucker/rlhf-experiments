import wandb
import os
import argparse
import datetime

wandb.login(key=os.environ["WANDB_API_KEY"])

parser = argparse.ArgumentParser()
parser.add_argument("path", type=str)
parser.add_argument("--project", type=str, default="tldr_summarize_pythia_will")
parser.add_argument("--name", type=str)
args = parser.parse_args()

wandb.init(project=args.project,
           tags = ["save_model"],)

artifact = wandb.Artifact(
    name = args.name,
    type = "model",
    description = f"Model saved on {datetime.datetime.now()}",
)

artifact.add_dir(args.path)
wandb.log_artifact(artifact)