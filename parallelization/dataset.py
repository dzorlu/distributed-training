import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, DistributedSampler
from transformers import PreTrainedTokenizer
import torch.distributed as dist
from typing import Optional
from .model.args import ModelArgs
from .logging import logger

def get_hf_dataloader(
    dataset_name: str,
    tokenizer: PreTrainedTokenizer,
    model_args: ModelArgs,
    batch_size: int,
    device_mesh,
):
    """
    Creates a data-parallel-aware DataLoader from a Hugging Face dataset.
    """
    seq_len = model_args.max_seq_len
    dataset = load_dataset(
        dataset_name,
        name=None, # Auto-selects config
        split="train"
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    def tokenize_and_format(examples):
        # Tokenize and truncate
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=seq_len,
        )
        
        # For next-token prediction, labels are inputs shifted by one.
        labels = [l[1:] + [tokenizer.pad_token_id] for l in tokenized["input_ids"]]
        
        return {
            "input_ids": tokenized["input_ids"],
            "labels": labels
        }

    # We need to set the format to torch and specify columns to keep
    formatted_dataset = dataset.map(tokenize_and_format, batched=True).with_format(
        type="torch", columns=["input_ids", "labels"]
    )

    # Get DP rank and size from the device mesh
    dp_rank = device_mesh.get_coordinate(dist.get_rank())[0]
    dp_size = device_mesh.get_dim_size("dp")

    logger.info(f"{dp_rank=}, {dp_size=}")


    sampler = DistributedSampler(
        formatted_dataset,
        num_replicas=dp_size,
        rank=dp_rank,
        shuffle=True
    )

    dataloader = DataLoader(
        formatted_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=2,
        pin_memory=True, # For faster CPU to GPU data transfer
    )

    return dataloader
