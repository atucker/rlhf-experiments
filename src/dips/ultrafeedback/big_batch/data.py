from dips.ultrafeedback.big_batch.config import Args
from dips.ultrafeedback.big_batch.tensor_ops import filter_by_length
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

def get_dataloaders(args: Args, tokenizer: AutoTokenizer):
    dataset = load_dataset(args.task.query_dataset, split="train")
    train_val_split = dataset.train_test_split(test_size=0.1, seed=args.seed) # use a consistent seed across runs
    dataset, validation_dataset = train_val_split["train"], train_val_split["test"]
    dataset = dataset.with_format("torch", columns=["instruction"])

    dataset = dataset.filter(filter_by_length,
                             fn_kwargs = {"tokenizer": tokenizer, "max_length": args.task.query_length})
    
    dataloader = DataLoader(dataset, batch_size=args.per_device_rollout_batch_size, shuffle=True)
    validation_dataset = validation_dataset.with_format("torch", columns=["instruction"])
    validation_dataset = validation_dataset.filter(filter_by_length,
                                                   fn_kwargs = {"tokenizer": tokenizer, 
                                                                  "max_length": args.task.query_length})
    validation_dataloader = DataLoader(validation_dataset, batch_size=args.per_device_eval_batch_size)

    return dataloader, validation_dataloader