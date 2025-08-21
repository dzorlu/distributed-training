from .llama2_model import Transformer
from .args import ModelArgs
import argparse
import os
import re


# Import the ray and profiler decorators
from parallelization import ray_distributed, profiler, flop_counter
from parallelization.profiler.decorator import step_profiler
from parallelization.profiler.utils import log_parameter_count
from .expert_parallel import ExpertParallel



import torch
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor.parallel import (
    ColwiseParallel, 
    RowwiseParallel, 
    parallelize_module,
    SequenceParallel,
    PrepareModuleInput,
    PrepareModuleInputOutput,
)
from torch.distributed.tensor import distribute_tensor, Shard, Replicate, Partial
from torch.distributed.fsdp import fully_shard





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
ffn_plan = {
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
}

# moe_ep_tp_plan = {
#     "layers.*.moe.experts": ExpertParallel(),
# }

moe_tp_plan = {
    # similar to FFN.
    "layers.*.feed_forward": PrepareModuleInputOutput(
        input_layouts=(Shard(1),),  # From ffn_norm
        desired_input_layouts=(Replicate(),),  # router need Replicate() - ALL-GATHER here
        use_local_input=True,
        use_local_output=True,
        # this is effectively reduce-scatter
        # data out of moe layer is different across the devices, so
        # it has to be reduced then scattered for upcoming norm layer
        # he _token_combine function handles the communication 
        # for Expert Parallelism. 
        # The output_layouts=(Partial(),) 
        # and desired_output_layouts=(Shard(1),) 
        # handle the communication for Tensor Parallelism, 
        # specifically completing the 
        # RowwiseParallel operation within the experts 
        # by performing a reduce-scatter. 
        # This two-level communication strategy 
        # is essential for combining EP and TP effectively.

        # these two are true regardless whether we do EP or not.
        output_layouts=(Partial(),),
        # back to norm
        desired_output_layouts=(Shard(1),),

    ),
}


def parallelize(model: nn.Module, world_mesh: DeviceMesh, model_args: ModelArgs, rank: int):
    """
        Parallelize for TP and (optionally) EP dimensions
    """
    
    ep_mesh = world_mesh['ep']
    ep_size = ep_mesh.size()
    tp_mesh = world_mesh['tp']
    tp_size = tp_mesh.size()

    # TP parallel
    parallelize_module(
        module=model,
        device_mesh=tp_mesh,
        parallelize_plan=tp_plan,
    )

    if model_args.use_moe:
        # Moe layers are both EP/TP.
        # Manually apply ExpertParallel to bypass validation in parallelize_module
        # This includes token dispath and token gather modules at the boundaries.
        expert_parallel_style = ExpertParallel()
        for layer in model.layers:
            expert_parallel_style._apply(layer.feed_forward.experts, world_mesh['ep','tp'])

        # moe input preparation goes into TP region.
        parallelize_module(
            module=model,
            device_mesh=tp_mesh,
            parallelize_plan=moe_tp_plan,
        )
    else:
        # If not MoE, parallelize FFN in TP
        parallelize_module(
            module=model,
            device_mesh=tp_mesh,
            parallelize_plan=ffn_plan,
        )




    






    