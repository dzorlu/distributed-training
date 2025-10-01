import torch
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor.parallel import (
    ColwiseParallel, 
    RowwiseParallel, 
    parallelize_module,
    SequenceParallel,
    PrepareModuleInput,
)
from torch.distributed.tensor import Shard, Replicate

from parallelization.logging import logger
from parallelization.model.args import ModelArgs


def parallelize(model: nn.Module, mesh: DeviceMesh, model_args: ModelArgs, rank: int):
    """
    Apply Tensor Parallelism to the model.
    
    This function applies TP across attention and feedforward layers using
    a combination of ColwiseParallel, RowwiseParallel, and SequenceParallel.
    """
    
    logger.info(f"Applying Tensor Parallelism on mesh {mesh}")
    
    tp_plan = {
        "tok_embeddings": RowwiseParallel(
            input_layouts=Replicate(),
            output_layouts=Shard(1),
        ),
        
        "layers.*.attention_norm": SequenceParallel(),
        
        "layers.*.attention": PrepareModuleInput(
            input_layouts=(Shard(1), Replicate()),
            desired_input_layouts=(Replicate(), Replicate()),
        ),
        
        "layers.*.attention.wq": ColwiseParallel(use_local_output=False),
        "layers.*.attention.wk": ColwiseParallel(use_local_output=False),
        "layers.*.attention.wv": ColwiseParallel(use_local_output=False),
        "layers.*.attention.wo": RowwiseParallel(output_layouts=Shard(1)),
        
        "layers.*.ffn_norm": SequenceParallel(),
        
        "layers.*.feed_forward": PrepareModuleInput(
            input_layouts=(Shard(1),),
            desired_input_layouts=(Replicate(),),
        ),
        
        "layers.*.feed_forward.w1": ColwiseParallel(),
        "layers.*.feed_forward.w3": ColwiseParallel(),
        "layers.*.feed_forward.w2": RowwiseParallel(output_layouts=Shard(1)),
        
        "norm": SequenceParallel(),
        
        "output": ColwiseParallel(
            input_layouts=Shard(1),
            output_layouts=Replicate()
        ),
    }
    
    parallelize_module(
        module=model,
        device_mesh=mesh,
        parallelize_plan=tp_plan
    )
    
    logger.info("Tensor Parallelism applied successfully")
