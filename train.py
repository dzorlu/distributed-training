
import argparse
import os
import re
import importlib
import wandb
from datetime import timedelta

# Import the ray and profiler decorators
from parallelization import ray_distributed
from parallelization.profiler.decorator import performance_monitor
from parallelization.profiler.utils import log_parameter_count
from parallelization.model.llama2_model import Transformer
from parallelization.model.llama2_model import ModelArgs
from parallelization.logging import logger, init_logger
from transformers import AutoTokenizer

from parallelization.utils import device_type, device_module
from torch._utils import _get_available_device_type, _get_device_module



import torch
import torch.nn.functional as F
from torch.distributed.tensor.debug import CommDebugMode
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh, DeviceMesh
from torch.distributed.tensor.parallel import (
    parallelize_module,
)
from torch.distributed.elastic.multiprocessing.errors import record
from torch.distributed.tensor import Shard, Replicate
from torch.distributed.fsdp import fully_shard
from torch.optim import Adam
from transformers import AutoTokenizer
from parallelization.dataset import get_hf_dataloader

def get_device_info() -> tuple[str, torch.device]:
    device_type = _get_available_device_type() or "cuda"
    device_module = _get_device_module(device_type)  # default device_module:torch.cuda
    logger.info(f"device_type: {device_type}, device_module: {device_module}")
    return device_type, device_module

def apply_parallelization(
    model: torch.nn.Module,
    model_name: str,
    world_mesh: DeviceMesh,
    model_args: ModelArgs,
    rank: int
):
    """Dynamically import and apply the parallelization plan from the model's directory."""
    try:
        module_path = f"parallelization.{model_name}.parallelize"
        parallelize_module_import = importlib.import_module(module_path)
        parallelize_fn = getattr(parallelize_module_import, "parallelize")
        # This function will now apply the parallelization in-place on the model
        parallelize_fn(model, world_mesh, model_args, rank)
    except (ModuleNotFoundError, AttributeError) as e:
        raise ImportError(f"Could not import or find 'parallelize' function in '{module_path}'. Make sure the module and function exist.") from e


def extract_unique_param_names(model):
    """Extract unique parameter patterns without layer numbers"""
    patterns = []
    
    for name, param in model.named_parameters():
        # Remove .weight, .bias suffix
        name_without_suffix = '.'.join(name.split('.')[:-1])
        
        # Remove layers.X. pattern if it exists
        pattern = re.sub(r'^layers\.\d+\.', '', name_without_suffix)
        
        patterns.append(pattern)
    
    # Get unique patterns and preserve order
    unique_patterns = []
    seen = set()
    for pattern in patterns:
        if pattern not in seen:
            unique_patterns.append(pattern)
            seen.add(pattern)
    
    return unique_patterns

@record
def main(args):
    # Flight recorder setup via environment variables
    os.environ["TORCH_NCCL_DUMP_ON_TIMEOUT"] = "1"
    os.environ["TORCH_FR_BUFFER_SIZE"] = "2000"
    os.environ["TORCH_FR_DUMP_TEMP_FILE"] = "/tmp/nccl_trace"
    
    init_logger()

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ.get("RANK", str(local_rank)))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    device_type, device_module = get_device_info()
    torch.cuda.set_device(local_rank)   # set by torchrun
    device = torch.device(f"{device_type}:{int(os.environ['LOCAL_RANK'])}")
    # Device has to be set before creating TorchFT manager.
    device_module.set_device(device)


    logger.info(f"device: {device}")
    logger.info(f"local_rank: {local_rank}, rank: {rank}, world_size: {world_size}")

    # Initialize the default PG with an explicit device mapping
    if dist.is_initialized():
        dist.destroy_process_group()
    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=60),
        device_id=local_rank,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)


    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    dp_size = args.dp_size
    tp_size = args.tp_size
    
    if args.use_moe:
        ep_size = args.ep_size
        if tp_size * ep_size * dp_size != world_size:
            raise ValueError(f"World size {world_size} must be equal to dp_size * tp_size * ep_size")
        
        device_mesh = init_device_mesh(
            "cuda",
            (dp_size, ep_size, tp_size),
            mesh_dim_names=("dp", "ep", "tp")
        )
        device_mesh['tp']
        device_mesh['ep','tp']
    else:
        if dp_size * tp_size != world_size:
            raise ValueError(f"World size {world_size} must be equal to dp_size * tp_size")
        device_mesh = init_device_mesh("cuda", (dp_size, tp_size), mesh_dim_names=("dp","tp",))

    logger.info(f"{device_mesh=}")
    assert torch.cuda.current_device() == local_rank

    rank = device_mesh.get_rank()
    if args.wandb:
        if rank == 0:
            wandb.init(project="distributed-training")

    logger.info(f"{tokenizer.vocab_size=}")
    model_args = ModelArgs(use_moe=args.use_moe, vocab_size=tokenizer.vocab_size)
    
    # This context allows you to define a model's architecture and a
    # ll its parameters without allocating any memory for the weights, 
    with torch.device("meta"):
        model = Transformer.from_model_args(model_args)
    log_parameter_count(model, model_args)


    # Dynamically apply parallelization plan
    apply_parallelization(
        model=model,
        model_name=args.model_name,
        world_mesh=device_mesh,
        model_args=model_args,
        rank=local_rank
    )
    
    logger.info("=== Parameter Analysis After Parallelization ===")
    regular_tensors = []
    dtensors = []

    for name, param in model.named_parameters():
        if hasattr(param, 'placements'):  # DTensor
            dtensors.append(name)
        else:  # Regular tensor
            regular_tensors.append(name)
            
    logger.info(f"DTensors: {len(dtensors)}")
    logger.info(f"Regular tensors: {len(regular_tensors)}")
    if regular_tensors:
        logger.info(f"{device=} Regular tensor parameters:")
        for name in regular_tensors:
            print(f"{device=}, {name=}")
            pass

    # logger.info(f"to_empty")
    model.to_empty(device=device)
    # The weights are initialized directly on each target GPU
    # after the model has been parallelized
    with torch.no_grad():
        logger.info(f"init_weights")
        model.init_weights()

    # foreach=False is not optimized.
    optimizer = Adam(model.parameters(), lr=args.lr, foreach=False)

    monitor = performance_monitor(
        model,
        enabled=args.profile,
        flop_counter_step=2,
        comm_logger_step=2,
    )

    @monitor
    def training_step(x, y):
        # Fake forward pass for now
        logits = model(x)
        logger.info(f"Logits shape: {logits.shape}, requires_grad: {logits.requires_grad}")
        logger.info(f"{y.shape=}")
        # Reshape for cross-entropy loss
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)), 
            y.view(-1),
            ignore_index=tokenizer.pad_token_id
        )
        logger.info(f"Loss shape: {loss.shape}, requires_grad: {loss.requires_grad}, {loss=}")
        loss.backward()
        return loss

    batch_size = 2

    # --- Data Loading ---
    dataloader = get_hf_dataloader(
        dataset_name=args.dataset_name,
        dataset_config_name=args.dataset_config_name,
        dataset_split=args.dataset_split,
        tokenizer=tokenizer,
        model_args=model_args,
        batch_size=batch_size,
        device_mesh=device_mesh,
    )

    for i, batch in enumerate(dataloader):

        x = batch['input_ids'].to(device, non_blocking=True)
        y = batch['labels'].to(device, non_blocking=True)
        loss = training_step(x, y)
        
        optimizer.step()

        if args.wandb and rank == 0:
            wandb.log({"loss": loss.item()})

    if args.wandb and rank == 0:
        wandb.finish()
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch Distributed Training")
    parser.add_argument("--use-moe", action="store_true", help="Whether to use MoE")
    parser.add_argument("--model-name", type=str, default="model", help="The name of the model folder under /parallelization")
    parser.add_argument("--tp-size", type=int, default=2, help="Tensor parallel size")
    parser.add_argument("--dp-size", type=int, default=1, help="Data parallel size")
    parser.add_argument("--ep-size", type=int, default=2, help="Expert parallel size for MoE models")
    parser.add_argument("--num-nodes", type=int, default=1, help="Number of nodes for distributed training")
    parser.add_argument("--gpus-per-node", type=int, default=2, help="Number of GPUs per node")
    parser.add_argument("--tokenizer-name", type=str, default="Qwen/Qwen-tokenizer", help="The name of the tokenizer to use")
    parser.add_argument("--dataset-name", type=str, default="wikitext", help="Hugging Face dataset name")
    parser.add_argument("--dataset-config-name", type=str, default="wikitext-2-raw-v1", help="Hugging Face dataset config name (e.g., 'en', 'wikitext-2-raw-v1')")
    parser.add_argument("--dataset-split", type=str, default="train", help="Dataset split to use (e.g., 'train', 'train[:1%]')")
    
    # Profiler arguments
    parser.add_argument("--profile", action="store_true", default=False, help="Enable PyTorch profiler")
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=True, help="Enable W&B logging by default (use --no-wandb to disable)")
    
    # Training arguments
    parser.add_argument("--lr", type=float, default=0.0005, help="Learning rate for the optimizer")
    
    args = parser.parse_args()
    
    # Apply decorators dynamically based on CLI args
    training_func = main
    
    # Apply ray decorator for distributed training
    distributed_main = ray_distributed(num_nodes=args.num_nodes, gpus_per_node=args.gpus_per_node)(training_func)
    distributed_main(args)