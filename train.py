
import argparse
import os
import re
import importlib

# Import the ray and profiler decorators
from parallelization import ray_distributed, profiler, flop_counter
from parallelization.profiler.decorator import step_profiler
from parallelization.profiler.utils import log_parameter_count
from parallelization.model.llama2_model import Transformer
from parallelization.model.llama2_model import ModelArgs
from parallelization.logging import logger, init_logger
from transformers import AutoTokenizer

from parallelization.utils import device_type, device_module


import torch
import torch.nn.functional as F
from torch.distributed.tensor.debug import CommDebugMode
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh, DeviceMesh
from torch.distributed.tensor.parallel import (
    parallelize_module,
)
from torch.distributed.tensor import Shard, Replicate
from torch.distributed.fsdp import fully_shard
from torch.optim import Adam
from transformers import AutoTokenizer
from parallelization.dataset import get_hf_dataloader


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

def main(args):
    init_logger()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)   # set by torchrun
    device = torch.device("cuda", local_rank)
    logger.info(f"device: {device}")
    
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
    else:
        if dp_size * tp_size != world_size:
            raise ValueError(f"World size {world_size} must be equal to dp_size * tp_size")
        device_mesh = init_device_mesh("cuda", (dp_size, tp_size), mesh_dim_names=("dp","tp",))


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
            #print(f"{device=}, {name=}")
            pass

    model.to_empty(device=device)
    # The weights are initialized directly on each target GPU
    # after the model has been parallelized
    with torch.no_grad():
        model.init_weights()

    # foreach=False is not optimized.
    optimizer = Adam(model.parameters(), lr=0.25, foreach=False)

    @flop_counter(model, enabled=args.profile, step_to_measure=2)
    def training_step(x, y, step_num=None):
        # https://docs.pytorch.org/tutorials/recipes/distributed_comm_debug_mode.html
        comm_mode = CommDebugMode()
        with comm_mode:
            # Fake forward pass for now
            logits = model(x)
            # Reshape for cross-entropy loss
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), 
                y.view(-1),
                ignore_index=tokenizer.pad_token_id
            )
        if dist.get_rank() == 0:
            comm_mode.log_comm_debug_tracing_table_to_file(
                noise_level=1,
                file_name=f"comm_debug_{step_num}.txt"
                )
            comm_mode.generate_json_dump(noise_level=2)
        loss.backward()
        return loss

    num_iter = 10
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
        if i >= num_iter:
            break

        x = batch['input_ids'].to(device, non_blocking=True)
        y = batch['labels'].to(device, non_blocking=True)
        loss = training_step(x, y, step_num=i)
        optimizer.step()

        # Step the profiler if profiling is enabled
        step_profiler()

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
    args = parser.parse_args()
    
    # Apply decorators dynamically based on CLI args
    training_func = main
    
    # Apply profiler decorator if requested
    if args.profile:
        training_func = profiler(enabled=True)(training_func)
    
    
    # Apply ray decorator for distributed training
    distributed_main = ray_distributed(num_nodes=args.num_nodes, gpus_per_node=args.gpus_per_node)(training_func)
    distributed_main(args)