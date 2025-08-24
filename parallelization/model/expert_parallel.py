
from functools import partial
from typing import Callable, Literal

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.tensor import (
    DeviceMesh,
    distribute_module,
    distribute_tensor,
    DTensor,
    Replicate,
    Shard,
)
from torch.distributed._functional_collectives import all_to_all_single_autograd
from torch.distributed.tensor.parallel import ParallelStyle
from torch.distributed.tensor.placement_types import Placement

from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    parallelize_module,
    PrepareModuleInput,
    PrepareModuleInputOutput,
    RowwiseParallel,
    SequenceParallel,
)

from parallelization.logging import logger


# # 4. FeedForward module needs input redistribution
# # Feed forward layers
# # return self.w2(F.silu(self.w1(x)) * self.w3(x))
# "layers.*.feed_forward.w1": ColwiseParallel(),
# "layers.*.feed_forward.w3": ColwiseParallel(),
# # Reduce-scatter op here for norm operations. 
# "layers.*.feed_forward.w2": RowwiseParallel(output_layouts=Shard(1)), 




class ExpertParallel(ParallelStyle):
    def __init__(self):
        super().__init__()
        pass

    def _partition_fn(self, name, module, device_mesh):
        """
        Applies to EP
        The weights are 3D (num_experts, dim, hidden_dim)
        """
        # the weights are 3D (num_experts, dim, hidden_dim)
        for name, param in module.named_parameters():
            dist_param = nn.Parameter(
                distribute_tensor(
                    param, device_mesh, [Shard(0)]
                )
            )
            logger.info(f"{name=}")
            module.register_parameter(name, dist_param)

    def _partition_2d_fn(self, name, module, device_mesh):
        """
        Applies to TP + EP
        The weights are 3D (num_experts, dim, hidden_dim)
        """
        # The TP applies to Shard(2).
        # This is confusing, because in ColwiseParallel it appleis to Shard(0).
        # https://github.com/pytorch/pytorch/blob/main/torch/distributed/tensor/parallel/style.py#L125C41-L125C51
        # but We use bmm and need to make sure that ColwiseParallel applies to outer dimension.
        # https://github.com/tgale96/grouped_gemm/blob/main/benchmark.py
        
        module.register_parameter(
            "w1",
            nn.Parameter(distribute_tensor(module.w1, device_mesh, [Shard(0), Shard(2)])),
        )  # Column-wise sharding

        # w2 shape = (experts, in_dim, out_dim)
        module.register_parameter(
            "w2",
            nn.Parameter(distribute_tensor(module.w2, device_mesh, [Shard(0), Shard(1)])),
        )  # Row-wise sharding

        # w3 shape = (experts, out_dim, in_dim)
        module.register_parameter(
            "w3",
            nn.Parameter(distribute_tensor(module.w3, device_mesh, [Shard(0), Shard(2)])),
        )  # Column-wise sharding


    def _token_dispatch(self, model, inputs, device_mesh):
        """
            All-to-all communication
            input_splits is different coming from each device (assuming some data parallelism)
        """
        ep_group = device_mesh.get_group("ep")
        ep_size = ep_group.size()

        x_gathered, num_tokens_per_expert = inputs
        num_tokens_per_expert_group = num_tokens_per_expert.new_empty(
            num_tokens_per_expert.shape[0]
        )

        # distributed transpose operation.
        # 0th GPU gets all 0th row

        # Preliminary all-to-all to exchange token counts. This is used to
        # calculate the split sizes for the main token all-to-all dispatch.
        #
        # Before (on GPU 0):
        #   `num_tokens_per_expert`: [10, 5, 12, 8, 11, 6, 13, 7]
        #   (Counts of local tokens for all 8 global experts)
        #
        # After (on GPU 0, which hosts experts 0 and 1):
        #   `num_tokens_per_expert_group` is filled with:
        #   [10, 5, | 9, 4, | 14, 2, | 3, 11]
        #   (Counts for my local experts [E0,E1] from GPU0, GPU1, GPU2, GPU3)
        
        dist.all_to_all_single(
            num_tokens_per_expert_group, # output!
            num_tokens_per_expert, # input
            group=ep_group,
        )

        logger.info(f"{num_tokens_per_expert=} {num_tokens_per_expert_group=}")
        input_splits = num_tokens_per_expert.view(
            ep_size, -1
        ).sum(dim=1).to(torch.device("cpu"))

        output_splits = num_tokens_per_expert_group.view(
            ep_size, -1
        ).sum(dim=1).to(torch.device("cpu"))

        self.input_splits = input_splits.tolist()
        self.output_splits = output_splits.tolist()

        # uneven communication

        # On GPU 0:
        # - Total tokens before send (sum of num_tokens_per_expert): 72
        # - input_splits (how to slice the 72 tokens for sending): [15, 20, 17, 20]
        # - output_splits (how many tokens to expect from each GPU): [15, 13, 16, 14]

        # Before all_to_all, each GPU has a different number of tokens and a different plan:
        # GPU 0: tensor of size 72, sends chunks of [15, 20, 17, 20]
        # GPU 1: (example) tensor of size 80, sends chunks of [13, 25, 22, 20]
        # GPU 2: (example) tensor of size 75, sends chunks of [16, 18, 21, 20]
        # GPU 3: (example) tensor of size 68, sends chunks of [14, 15, 19, 20]

        # After all_to_all on GPU 0:
        # - Receives: 15 from GPU0, 13 from GPU1, 16 from GPU2, 14 from GPU3
        # - Output tensor size = sum(output_splits) = 15 + 13 + 16 + 14 = 58
        # - This new tensor of 58 tokens contains data for GPU 0's local experts (E0, E1),
        #   but is grouped by source GPU, not by expert ID. It needs a local shuffle.

        # all_to_all_single_autograd allows differentiable data transfer
        logger.info(f"{self.output_splits=} {self.input_splits=}")

        x_gathered = all_to_all_single_autograd(
            x_gathered,
            self.output_splits,
            self.input_splits,
            ep_group,
        )

        # num_tokens_per_expert_group
        #   [10, 5, | 9, 4, | 14, 2, | 3, 11]
        # 
        #   x_gathered on GPU 0 (shape: [58, h])
        #  +------------------------------------------------+
        #  |                                                |
        #  |  Block of 15 tokens RECEIVED from GPU 0        |
        #  |  (Contains 10 tokens for MY E0, 5 for MY E1)   |
        #  |                                                |
        #  +------------------------------------------------+  <-- Boundary at index 14
        #  |                                                |
        #  |  Block of 13 tokens RECEIVED from GPU 1        |
        #  |  (Contains 9 tokens for MY E0, 4 for MY E1)    |
        #  |                                                |
        #  +------------------------------------------------+  <-- Boundary at index 27 (14+13)
        #  |                                                |
        #  |  Block of 16 tokens RECEIVED from GPU 2        |
        #  |  (Contains 14 tokens for MY E0, 2 for MY E1)   |
        #  |                                                |
        #  +------------------------------------------------+  <-- Boundary at index 43 (27+16)
        #  |                                                |
        #  |  Block of 14 tokens RECEIVED from GPU 3        |
        #  |  (Contains 3 tokens for MY E0, 11 for MY E1)   |
        #  |                                                |
        #  +------------------------------------------------+  <-- Final boundary at index 57

        #   Target layout for x_gathered (shape: [58, h])
        #  +------------------------------------------------+
        #  |                                                |
        #  |  All 36 tokens for MY Expert 0                 |
        #  |  (Gathered from the 4 blocks above)            |
        #  |                                                |
        #  +------------------------------------------------+  <-- Boundary at index 35
        #  |                                                |
        #  |  All 22 tokens for MY Expert 1                 |
        #  |  (Gathered from the 4 blocks above)            |
        #  |                                                |
        #  +------------------------------------------------+ 

        # target for num_tokens_per_expert_group
        #    [36, 22]


        # Reshape to see GPU-expert structure
        tokens = num_tokens_per_expert_group.view(-1, ep_size)  
        # Shape: [4, 2] where dim0=GPU, dim1=expert
        # [[10,  5],  <- GPU 0: 10 tokens for E0, 5 for E1
        #  [ 9,  4],  <- GPU 1: 9 tokens for E0, 4 for E1
        #  [14,  2],  <- GPU 2: 14 tokens for E0, 2 for E1
        #  [ 3, 11]]  <- GPU 3: 3 tokens for E0, 11 for E1
        expert_per_device = num_tokens_per_expert_group.shape[0] // ep_size
        expert_ids = torch.repeat_interleave(
            torch.arange(expert_per_device).repeat(ep_size).to('cuda'),  # [0, 1, 0, 1, 0, 1, 0, 1] - expert pattern
            num_tokens_per_expert_group  # [10,5,9,4,14,2,3,11] - repeat counts
        )
        
        # index looks like
        # tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 15, 16, 17, 18, 19, 20, 21, 22,
        # 23, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 44, 45, 46,
        # 10, 11, 12, 13, 14, 24, 25, 26, 27, 42, 43, 47, 48, 49, 50, 51, 52, 53,
        # 54, 55, 56, 57])
        self.index = torch.argsort(expert_ids, stable=True)
        x_reorganized = x_gathered[self.index, :]

        # per expert aggregation
        num_tokens_per_expert_group_agg = tokens.sum(dim=1)

        return x_reorganized, num_tokens_per_expert_group_agg


    def _token_combine(self, mod, routed_output, device_mesh):
        """
            Reverse the _token_dispatch operation.
            All-to-all to combine the output
        """
        ep_group = device_mesh.get_group("ep")
        ep_size = ep_group.size()

        reverse_ix = self.index.argsort()
        routed_output = routed_output[reverse_ix,:]

        # reversing the op
        routed_output = all_to_all_single_autograd(
            routed_output,
            self.input_splits,
            self.output_splits,
            ep_group,
        )

        return routed_output


    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        _partition_fn = self._partition_2d_fn
        return distribute_module(
            module,
            device_mesh,
            partition_fn=_partition_fn,
            input_fn=self._token_dispatch,
            output_fn=self._token_combine,
        )