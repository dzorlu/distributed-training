
import os
import sys
import torch
import torch.nn as nn
import logging

from torch.distributed._tensor import Shard, DTensor

from torch.distributed._tensor.device_mesh import init_device_mesh

from torch.distributed.tensor.parallel import (
    parallelize_module,
    ColwiseParallel,
    RowwiseParallel,
)


def rank_log(_rank, logger, msg):
    """helper function to log only on global rank 0"""
    if _rank == 0:
        logger.info(f" {msg}")



# SETUP: 2 ranks, global dims: [B=4, S=1024, H=512, D=2048]
# Sequence Parallel Config:
# "in_proj": ColwiseParallel(input_layouts=Shard(1), output_layouts=Shard(-1))  # defaults shown
# "out_proj": RowwiseParallel(input_layouts=Shard(-1), output_layouts=Shard(1))

# ==================== ColwiseParallel ====================
# class ColwiseParallel:
#     def __init__(self, input_layouts=None, output_layouts=None):
#         self.input_layouts = input_layouts or Replicate()      # What we receive
#         self.desired_input_layouts = Replicate()               # What we need (ALWAYS)
#         self.output_layouts = output_layouts or Shard(-1)      # What we produce
    
    # WEIGHT SHARDING (_partition_linear_fn)
    # Original: Linear(512, 2048) → weight [2048, 512]
    # Shard(0) on weight → each rank gets [1024, 512]
    # Rank 0: weight[0:1024, :]     # first half of output features
    # Rank 1: weight[1024:2048, :]  # second half of output features
    
    # INPUT TRANSFORMATION (_prepare_input_fn)
    # Input arrives: [4, 512, 512] with Shard(1) placement
    # Target: Replicate() (desired_input_layouts)
    # Operation: ALL-GATHER along dim 1
    # After: [4, 1024, 512] replicated on all ranks
    
    # COMPUTATION
    # input @ weight.T = [4, 1024, 512] @ [512, 1024] = [4, 1024, 1024]
    # Each rank computes different output features
    
    # OUTPUT TRANSFORMATION (_prepare_output_fn)  
    # Current: [4, 1024, 1024] with implicit Shard(-1) placement
    # Target: Shard(-1) (output_layouts - same as current)
    # Operation: NO-OP (already feature-sharded)
    # Final: [4, 1024, 1024] feature-sharded

# State after ColwiseParallel:
# Rank 0: [4, 1024, 1024] - features[0:1024]
# Rank 1: [4, 1024, 1024] - features[1024:2048]

# ==================== RowwiseParallel ====================
# class RowwiseParallel:
#     def __init__(self, input_layouts=None, output_layouts=None):
#         self.input_layouts = input_layouts or Shard(-1)        # What we receive
#         self.desired_input_layouts = Shard(-1)                 # What we need (ALWAYS)
#         self.output_layouts = output_layouts or Replicate()    # What we produce
    
    # WEIGHT SHARDING (_partition_linear_fn)
    # Original: Linear(2048, 512) → weight [512, 2048]
    # Shard(1) on weight → each rank gets [512, 1024]
    # Rank 0: weight[:, 0:1024]     # first half of input features
    # Rank 1: weight[:, 1024:2048]  # second half of input features
    
    # INPUT TRANSFORMATION (_prepare_input_fn)
    # Input arrives: [4, 1024, 1024] with Shard(-1) placement
    # Target: Shard(-1) (desired_input_layouts - same)
    # Operation: NO-OP (already feature-sharded correctly)
    # After: [4, 1024, 1024] unchanged
    
    # COMPUTATION
    # input @ weight.T = [4, 1024, 1024] @ [1024, 512] = [4, 1024, 512]
    # Each rank computes partial output (needs reduction)
    # Placement after matmul: Partial() (implicit, needs reduction)
    
    # OUTPUT TRANSFORMATION (_prepare_output_fn)
    # Current: [4, 1024, 512] with Partial() placement
    # Target: Shard(1) (output_layouts we specified)
    # Operation: REDUCE-SCATTER
    #   1. Sum partial results across ranks
    #   2. Scatter along dim 1 (sequence)
    # Final: [4, 512, 512] sequence-sharded

# Final state after RowwiseParallel:
# Rank 0: [4, 512, 512] - seq[0:512], fully summed
# Rank 1: [4, 512, 512] - seq[512:1024], fully summed



logger = logging.getLogger(__name__)

class ToyModel(nn.Module):
    """MLP based model"""

    def __init__(self, in_features, out_features, rank):
        super().__init__()
        self.in_proj = nn.Linear(in_features, out_features)
        self.relu = nn.ReLU()
        self.out_proj = nn.Linear(out_features, 5)
        self.rank = rank
        print(f"model at rank: {self.rank}")


    def forward(self, x):

        def get_placement(tensor):
            if isinstance(tensor, DTensor):
                return tensor.placements[0]
            return "Local Tensor"

        print(f"[{self.rank}] {x.shape} before in_proj - {get_placement(x)}")
        x = self.in_proj(x)
        print(f"[{self.rank}] {x.shape} before relu" - {get_placement(x)}")
        x = self.relu(x)
        print(f"[{self.rank}] {x.shape} before out_proj" - {get_placement(x)}")
        out = self.out_proj(x)
        print(f"[{self.rank}] {out.shape} before out_proj" - {get_placement(out)}")
        return out


if __name__ == "__main__":
    _world_size = int(os.environ["WORLD_SIZE"])
    device_type = torch.accelerator.current_accelerator().type
    device_mesh = init_device_mesh(device_type=device_type, mesh_shape=(_world_size,))

    _rank = device_mesh.get_rank()
    print(f"Starting PyTorch Sequence Parallel example on rank {_rank}.")
    rank_log(_rank, logger, f"Device Mesh created: {device_mesh=}")

    in_features = 256
    out_features = 256
    model = ToyModel(
        in_features=in_features, 
        out_features=out_features,
        rank = _rank
    ).to(device_type)

    # Custom parallelization plan for the model
    sp_model = parallelize_module(
        module=model,
        device_mesh=device_mesh,
        parallelize_plan={
            # This takes a sharded tensor along the sequence dimension. SP -> TP region.
            #         
            # `prepare_input_fn`[https://github.com/pytorch/pytorch/blob/main/torch/distributed/tensor/parallel/style.py#L99] is 
            # tasked to conver input_layout to desired_input_layout.
            # `desired_input_layout` for ColWiseParallel is `Replicate`[https://github.com/pytorch/pytorch/blob/main/torch/distributed/tensor/parallel/style.py#L95]
            # So Shard(1) -> Replicate is an **all-gather** operation.
            # 
            # Second, weight sharding. By default, ColwiseParallel shards by Shard(0). Remember, the matmul operation is x * w.T
            #
            # Third, output_layout for ColwiseParallel is by default `Shard(-1)`. This is accomplished by weight sharding.

            "in_proj": ColwiseParallel(
                input_layouts=Shard(1),
                use_local_output=False
                ),

            # out_proj here refers to the `TP->SP` region.

            # prepare_input_fn is by default (Shard(-1)). And this is satisfied.

            # By default, the weight is sharded in Shard(1). [https://github.com/pytorch/pytorch/blob/main/torch/distributed/tensor/parallel/style.py#L250]
            # This projects the tensor back to its original `b,s,h`

            # THird, output_layout here is given as `Shard(1)`. We want to shard it back to sequence dimension. 
            # From `Shard(-1)` to `Shard(1)` is a reduce-scatter operation.

            "out_proj": RowwiseParallel(
                output_layouts=Shard(1),
                use_local_output=False
                ),
        },
    )


    # Create a optimizer for the parallelized module.
    lr = 0.25
    optimizer = torch.optim.AdamW(sp_model.parameters(), lr=lr, foreach=True)


    # Perform a num of iterations of forward/backward
    # and optimizations for the sharded module.
    num_iters = 10

    for i in range(num_iters):
        # For SP, input can be different across all ranks.
        # This, in addition to `ColwiseParallel(input_layouts=Shard(1))` 
        # simulates SP->TP region, where the input is sharded along the dimension 1. 
        # Input is [32, 512, in_features]
        inp = torch.rand(32, 512, in_features,  device=device_type)
        output = sp_model(inp)
        output.sum().backward()
        optimizer.step()