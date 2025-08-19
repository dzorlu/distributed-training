from parallelization.model.llama2_model import Transformer
from parallelization.model.moe import MoE
from parallelization.model.llama2_model import ModelArgs
import argparse
import os
import re
import importlib

# Import the ray and profiler decorators
from parallelization import ray_distributed, profiler, flop_counter
from parallelization.profiler.decorator import step_profiler
from parallelization.profiler.utils import log_parameter_count
from torch.distributed.tensor.debug import CommDebugMode
import torch.distributed as dist


import torch
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import (
    parallelize_module,
)
from torch.distributed.tensor import Shard, Replicate
from torch.distributed.fsdp import fully_shard
from torch.optim import Adam

def get_plan(plan_name: str):
    """Dynamically import and return the parallelization plan."""
    try:
        module = importlib.import_module(f"parallelization.{plan_name}.{plan_name}")
        plan_variable_name = f"{plan_name}_plan"
        return getattr(module, plan_variable_name)
    except (ModuleNotFoundError, AttributeError) as e:
        raise ImportError(f"Could not import plan '{plan_name}'. Make sure the plan exists and the naming convention is followed.") from e


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
    
    # For now, let's assume a 1D mesh for simplicity, suitable for EP.
    # We can add more complex mesh logic later.
    device_mesh = init_device_mesh("cuda", (world_size,))
    dp_rank = 0 # Assuming no data parallelism for now.

    model_args = ModelArgs()
    if args.use_moe:
        # PoC
        model = MoE.from_model_args(model_args=model_args)
    else:
        model = Transformer.from_model_args(model_args)
    log_parameter_count(model, model_args)
    #model.init_weights()


    param_names = extract_unique_param_names(model)
    print(param_names)

    # Dynamically get the parallelization plan
    parallelization_plan = get_plan('ep')

    model = parallelize_module(
        module=model,
        device_mesh=device_mesh,
        parallelize_plan=parallelization_plan
    )

    print("=== Parameter Analysis ===")
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
        # seeding with dp_rank to ensure identical inputs for TP groups
        torch.manual_seed(i + dp_rank)
        #x = torch.randint(model_args.vocab_size, (batch_size, seq_len)).to(device)
        x = torch.rand(batch_size, seq_len, model_args.dim).to(device)
        loss = training_step(x, step_num=i)
        optimizer.step()

        # Step the profiler if profiling is enabled
        step_profiler()

    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch Distributed Training")
    parser.add_argument("--use-moe", action="store_true", help="Whether to use MoE")
    parser.add_argument("--num-nodes", type=int, default=1, help="Number of nodes for distributed training")
    parser.add_argument("--gpus-per-node", type=int, default=2, help="Number of GPUs per node")
    parser.add_argument("--fsdp-enable", action="store_true", help="Enable FSDP with 2D parallelism")
    
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