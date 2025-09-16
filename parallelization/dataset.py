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

    if "wikitext" in dataset_name:
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
        shuffle = True
        drop_last = True
        num_workers = 2

    else:  # Handles zai-org/LongAlign-10k and potentially others with similar format
        def tokenize_and_format(examples):
            texts = []
            for messages_pair in examples["messages"]:
                # messages_pair is [user_dict, assistant_dict]
                if len(messages_pair) == 2:
                    user_msg = messages_pair[0]
                    assistant_msg = messages_pair[1]
                    
                    # Extract content from each message dict
                    user_content = user_msg.get("content", "") if isinstance(user_msg, dict) else str(user_msg)
                    assistant_content = assistant_msg.get("content", "") if isinstance(assistant_msg, dict) else str(assistant_msg)
                    
                    # Format as conversation
                    full_text = f"User: {user_content}\n\nAssistant: {assistant_content}"
                    texts.append(full_text)
                else:
                    texts.append("")
            
            tokenized = tokenizer(
                texts,
                truncation=True,
                padding="max_length",
                max_length=seq_len,
            )
            
            labels = [l[1:] + [tokenizer.pad_token_id] for l in tokenized["input_ids"]]
            
            return {
                "input_ids": tokenized["input_ids"],
                "labels": labels
            }

        # Apply mapping and format
        formatted_dataset = dataset.map(
            tokenize_and_format, 
            batched=True,
            batch_size=model_args.batch_size * 20,
            remove_columns=dataset.column_names,  # Remove original columns
            num_proc=20,
        ).with_format(
            type="torch", 
            columns=["input_ids", "labels"],
        )
        shuffle = False
        drop_last = True
        num_workers = 0

    # Get DP rank and size from the device mesh
    dp_rank = device_mesh.get_coordinate()[0]
    dp_size = device_mesh.mesh.size(0)

    logger.info(f"{dp_rank=}, {dp_size=}")

    sampler = DistributedSampler(
        formatted_dataset,
        num_replicas=dp_size,
        rank=dp_rank,
        shuffle=shuffle,
        drop_last=drop_last,
    )

    dataloader = DataLoader(
        formatted_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last,
    )
    logger.info(f"{dataloader=}")

    return dataloader