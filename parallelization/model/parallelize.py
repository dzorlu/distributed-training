from .llama2_model import Transformer
from .args import ModelArgs
import argparse
import os
import re


# Import the ray and profiler decorators
from .expert_parallel import ExpertParallel
from parallelization.logging import logger



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




def parallelize(model: nn.Module, mesh: DeviceMesh, model_args: ModelArgs, rank: int):
    """
        Parallelize  EP dimensions
    """
    
    logger.info(f"Parallelizing model on mesh {mesh}")
    expert_parallel_style = ExpertParallel()
    # Apply EP to all expert modules at once
    parallelize_module(
        module=model,
        device_mesh=mesh,
        parallelize_plan={
            "layers.*.feed_forward.experts": ExpertParallel(),
        }
    )





    






    