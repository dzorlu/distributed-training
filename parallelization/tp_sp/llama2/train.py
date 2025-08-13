from .llama2_model import Transformer
from .llama2_model import ModelArgs
import argparse
import os
import re


# Import the ray and profiler decorators
from parallelization import ray_distributed, profiler, flop_counter
from parallelization.profiler.decorator import step_profiler
from parallelization.profiler.utils import log_parameter_count


import torch
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import (
    ColwiseParallel, 
    RowwiseParallel, 
    parallelize_module,
    SequenceParallel,
    PrepareModuleInput,
)
from torch.distributed.tensor import distribute_tensor, Shard, Replicate
from torch.distributed.fsdp import fully_shard

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
    tp_size = 2
    if args.fsdp_enable:
        dp_size = world_size // tp_size
        device_mesh = init_device_mesh("cuda", (dp_size, tp_size), mesh_dim_names=("dp","tp"))
        tp_mesh = device_mesh["tp"]
        # to ensure TP gets the same data.
        dp_mesh = device_mesh["dp"]
        dp_rank = dp_mesh.get_local_rank()
    else:
        tp_mesh = init_device_mesh("cuda", (world_size,)) # 1D - TP.
    
    



    model_args = ModelArgs()
    model = Transformer.from_model_args(model_args)
    log_parameter_count(model, model_args)


    param_names = extract_unique_param_names(model)
    print(param_names)


    tp_plan = {
        # === Embeddings ===
        # Transferring lower payload embedding dimension (vs ~40k dimensional payload)
        "tok_embeddings": RowwiseParallel(
            # Token-ids are replicated in each GPU!
            input_layouts=Replicate(),
            # The token embeddings output Shard(1) to maintain consistent input format for all transformer layers.
            # Now EVERY transformer block receives the same format. 
            # the first operation in transformer block is `attention_norm` op which is SP.
            output_layouts=Shard(1),
        ),
        
        # === For each TransformerBlock (order follows forward pass) ===
        


        # 1. First operation: self.attention_norm(x)
        "layers.*.attention_norm": SequenceParallel(),  # Expects Shard(1), outputs Shard(1)

        # 2. Attention module needs input redistribution
        # PrepareModuleInput sees:
        # - Arg 1: DTensor [32, 128, 2048] with Shard(1)
        # - Arg 2: Convert to DTensor [256, 32] with Replicate()

        # After redistribution:
        # - Arg 1: DTensor [32, 256, 2048] with Replicate() 
        # - Arg 2: DTensor [256, 32] with Replicate()

        # Both are now DTensors with consistent global shapes!
        "layers.*.attention": PrepareModuleInput(
            input_layouts=(Shard(1), Replicate()),  # norm output is Shard(1), freqs_cis is 
            desired_input_layouts=(Replicate(), Replicate()),  # wq/wk/wv need Replicate() - ALL-GATHER here
        ),



        # Attention layers  
        # Unlike a regular tensor, a DTensor is aware of the parallelism plans and 
        # will automatically handle changes in the num_heads dimension.
        # The use_local_output=False ensures you get tensors with **global shapes**, 
        # making view operations work correctly without manual num_heads adjustment.
        "layers.*.attention.wq": ColwiseParallel(use_local_output=False),
        "layers.*.attention.wk": ColwiseParallel(use_local_output=False),
        "layers.*.attention.wv": ColwiseParallel(use_local_output=False),
        # Reduce-scatter op here. TP -> SP
        "layers.*.attention.wo": RowwiseParallel(output_layouts=Shard(1)),
        
        # 3. Second operation: self.ffn_norm(h)
        "layers.*.ffn_norm": SequenceParallel(),  # Expects Shard(1), outputs Shard(1)
        
        # 4. FeedForward module needs input redistribution
        "layers.*.feed_forward": PrepareModuleInput(
            input_layouts=(Shard(1),),  # From ffn_norm
            desired_input_layouts=(Replicate(),),  # w1/w3 need Replicate() - ALL-GATHER here
        ),
        
        # Feed forward layers
        # return self.w2(F.silu(self.w1(x)) * self.w3(x))
        "layers.*.feed_forward.w1": ColwiseParallel(),
        "layers.*.feed_forward.w3": ColwiseParallel(),
        # Reduce-scatter op here for norm operations. 
        "layers.*.feed_forward.w2": RowwiseParallel(output_layouts=Shard(1)), 
        
        # === Final model operations ===
        # Norms - all SequenceParallel
        "norm": SequenceParallel(),  # Final norm before output
        
        # Final output layer
        "output": ColwiseParallel(
            input_layouts=Shard(1),  # From final norm (needs to be specified!)
            # so that we don't have to fetch from other ranks. 
            # it is replicated in each GPU
            # for loss calculation
            output_layouts=Replicate()
        ),
    }


    model = parallelize_module(
        module=model,
        device_mesh=tp_mesh,
        parallelize_plan=tp_plan
    )

    if args.fsdp_enable:
        model = fully_shard(model, mesh=device_mesh)


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

    lr = 0.25
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, foreach=True)

    @flop_counter(model, enabled=args.profile, step_to_measure=1)
    def training_step(x, step_num=None):
        loss = model(x).sum()
        loss.backward()
        return loss

    num_iter = 10
    batch_size = 2
    seq_len = 128
    for i in range(num_iter):
        # seeding with dp_rank to ensure identical inputs for TP groups
        torch.manual_seed(i + dp_rank)
        x = torch.randint(model_args.vocab_size, (batch_size, seq_len)).to(device)
        loss = training_step(x, step_num=i)
        optimizer.step()

        # Step the profiler if profiling is enabled
        step_profiler()

    torch.distributed.destroy_process_group()




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch TP/SP example")
    parser.add_argument("--num-nodes", type=int, default=1, help="Number of nodes for distributed training")
    parser.add_argument("--gpus-per-node", type=int, default=1, help="Number of GPUs per node")
    
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