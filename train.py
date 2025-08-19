
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


import torch
from torch.distributed.tensor.debug import CommDebugMode
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh, DeviceMesh
from torch.distributed.tensor.parallel import (
    parallelize_module,
)
from torch.distributed.tensor import Shard, Replicate
from torch.distributed.fsdp import fully_shard
from torch.optim import Adam


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

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)   # set by torchrun
    device = torch.device("cuda", local_rank)
    print(f"device: {device}")        # pin this process to its GPU

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    if args.use_moe:
        # 2D mesh for TP and EP
        if args.tp_size * args.ep_size != world_size:
            raise ValueError(f"World size {world_size} must be equal to tp_size * ep_size")
        
        device_mesh = init_device_mesh(
            "cuda",
            (args.ep_size, args.tp_size),
            mesh_dim_names=("ep", "tp")
        )
    else:
        # 1D mesh for TP
        if world_size != args.tp_size:
            raise ValueError(f"World size {world_size} must be equal to tp_size {args.tp_size} for non-MoE models")
        device_mesh = init_device_mesh("cuda", (args.tp_size,), mesh_dim_names=("tp",))


    model_args = ModelArgs(use_moe=args.use_moe)
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
    
    print("=== Parameter Analysis After Parallelization ===")
    regular_tensors = []
    dtensors = []

    for name, param in model.named_parameters():
        if hasattr(param, 'placements'):  # DTensor
            dtensors.append(name)
        else:  # Regular tensor
            regular_tensors.append(name)
            
    print(f"DTensors: {len(dtensors)}")
    print(f"Regular tensors: {len(regular_tensors)}")
    if regular_tensors:
        print("Regular tensor parameters:")
        for name in regular_tensors:
            print(f"  {name}")

    model = model.to(device) # need to do this still.

    # foreach=False is not optimized.
    optimizer = Adam(model.parameters(), lr=0.25, foreach=False)

    @flop_counter(model, enabled=args.profile, step_to_measure=2)
    def training_step(x, step_num=None):
        comm_mode = CommDebugMode()
        with comm_mode:
            # Fake forward pass for now
            loss = model(x).sum()
        if dist.get_rank() == 0:
            print(f"------------- COMM DEBUG MODE: Step {step_num} -------------")
            comm_mode.log_comm_debug_tracing_table_to_file(file_name=f"comm_debug_{step_num}.txt")
        loss.backward()
        return loss

    num_iter = 10
    batch_size = 2
    seq_len = 128
    for i in range(num_iter):
        # seeding with rank to ensure identical inputs for TP groups
        torch.manual_seed(i + dist.get_rank())
        x = torch.randint(0, model_args.vocab_size, (batch_size, seq_len), device=device)
        loss = training_step(x, step_num=i)
        optimizer.step()

        # Step the profiler if profiling is enabled
        step_profiler()

    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch Distributed Training")
    parser.add_argument("--use-moe", action="store_true", help="Whether to use MoE")
    parser.add_argument("--model-name", type=str, default="model", help="The name of the model folder under /parallelization")
    parser.add_argument("--tp-size", type=int, default=2, help="Tensor parallel size")
    parser.add_argument("--ep-size", type=int, default=2, help="Expert parallel size for MoE models")
    parser.add_argument("--num-nodes", type=int, default=1, help="Number of nodes for distributed training")
    parser.add_argument("--gpus-per-node", type=int, default=2, help="Number of GPUs per node")
    
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