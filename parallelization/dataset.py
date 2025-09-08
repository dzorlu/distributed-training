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
    dataset_config_name: Optional[str],
    dataset_split: str,
    tokenizer: PreTrainedTokenizer,
    model_args: ModelArgs,
    device_mesh,
):
    """
    Creates a data-parallel-aware DataLoader from a Hugging Face dataset.
    """
    seq_len = model_args.max_seq_len
    batch_size = model_args.batch_size
    
    dataset = load_dataset(
        dataset_name,
        name=dataset_config_name,
        split=dataset_split
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    def tokenize_and_format(examples):
        # Handle different dataset formats
        texts = []
        
        # Check what columns we have
        if "text" in examples:
            texts = examples["text"]
        elif "messages" in examples:
            # Handle conversation format (like LongAlign)
            for messages in examples["messages"]:
                if isinstance(messages, list):
                    # Concatenate all messages in the conversation
                    conversation = ""
                    for msg in messages:
                        if isinstance(msg, dict):
                            # Format: role: content
                            role = msg.get("role", "")
                            content = msg.get("content", "")
                            conversation += f"{role}: {content}\n"
                        else:
                            conversation += str(msg) + "\n"
                    texts.append(conversation.strip())
                else:
                    # Fallback: convert to string
                    texts.append(str(messages))
        else:
            raise ValueError(f"No text column found in dataset: {examples.keys()}")
        
        # Ensure texts is a list of strings
        if not isinstance(texts, list):
            texts = [texts]
        texts = [str(t) for t in texts]  # Convert everything to strings
        
        # Tokenize
        tokenized = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=seq_len,
        )
        
        # For next-token prediction, labels are inputs shifted by one
        labels = [l[1:] + [tokenizer.pad_token_id] for l in tokenized["input_ids"]]
        
        return {
            "input_ids": tokenized["input_ids"],
            "labels": labels
        }

    # Apply mapping and format
    formatted_dataset = dataset.map(
        tokenize_and_format, 
        batched=True,
        remove_columns=dataset.column_names  # Remove original columns
    ).with_format(
        type="torch", 
        columns=["input_ids", "labels"]
    )

    # Get DP rank and size from the device mesh
    dp_rank = device_mesh.get_coordinate()[0]
    dp_size = device_mesh.mesh.size(0)

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
        pin_memory=True,
    )
    logger.info(f"{dataloader=}")

    return dataloader