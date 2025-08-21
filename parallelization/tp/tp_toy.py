# Simple replication of https://docs.pytorch.org/tutorials/intermediate/TP_tutorial.html

from curses import keyname
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
from ..logging import logger, init_logger

from torch.distributed._tensor.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import (
    parallelize_module,
    ColwiseParallel,
    RowwiseParallel,
)
from torch.distributed.tensor import distribute_tensor, Shard, Replicate


class MHA(nn.Module):
    def __init__(self, hidden_dim, nb_heads, device_mesh):
        super(MHA, self).__init__()
        
        # BEFORE PARALLELIZATION:
        # Each linear layer has weight shape [out_features, in_features] = [4096, 4096]
        # These are regular nn.Linear modules with regular tensors
        self.q = nn.Linear(hidden_dim, hidden_dim)  # Weight: [4096, 4096]
        self.k = nn.Linear(hidden_dim, hidden_dim)  # Weight: [4096, 4096]
        self.v = nn.Linear(hidden_dim, hidden_dim)  # Weight: [4096, 4096]
        self.o = nn.Linear(hidden_dim, hidden_dim)  # Weight: [4096, 4096]
        
        self.nb_heads = nb_heads
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // nb_heads  # 4096 / 8 = 512
        self.device_mesh = device_mesh

    def forward(self, x):
        """
        inputs:
            x: [batch, L, hidden_dim] = [32, 512, 4096]
               Initially a regular tensor, gets converted to Replicated DTensor
        """
        b, L = int(x.shape[0]), int(x.shape[1])

        # AFTER PARALLELIZATION WITH ColwiseParallel(use_local_output=False):
        # 
        # WEIGHT SHARDING (ColwiseParallel):
        # - Original weight: [4096, 4096]
        # - GPU 0 gets: weight[:2048, :] → shape [2048, 4096] (first half of ROWS)
        # - GPU 1 gets: weight[2048:, :] → shape [2048, 4096] (second half of ROWS)
        # - This is Shard(dim=0) - sharding dimension 0 (rows)
        #
        # COMPUTATION:
        # - Input x is replicated: each GPU has full [32, 512, 4096]
        # - GPU 0: computes x @ weight_gpu0.T → produces columns 0-2047 of output
        # - GPU 1: computes x @ weight_gpu1.T → produces columns 2048-4095 of output
        # - Hence "Colwise" - each GPU computes different OUTPUT COLUMNS
        #
        # OUTPUT with use_local_output=False:
        # - Returns DTensor with logical shape [32, 512, 4096]
        # - Physically stored as [32, 512, 2048] per GPU
        # - Has placement Shard(dim=2) - sharded on last dimension
        
        q = self.q(x)  # DTensor: logical [32, 512, 4096], physical [32, 512, 2048] per GPU
        k = self.k(x)  # DTensor: logical [32, 512, 4096], physical [32, 512, 2048] per GPU
        v = self.v(x)  # DTensor: logical [32, 512, 4096], physical [32, 512, 2048] per GPU

        rank = device_mesh.get_rank()

        # INSPECTING WEIGHTS AND OUTPUTS:
        weight_q = self.q.weight  # This is the Linear module's weight (DTensor)
        
        # q.to_local() shows the LOCAL data this GPU holds
        logger.info(f"[Rank {rank}] Full local tensor q info: {q.to_local().shape}")
        # Output: [32, 512, 2048] - each GPU has half the hidden_dim
        
        # weight.to_local() shows the LOCAL weight shard this GPU holds  
        logger.info(f"[Rank {rank}] Full local weight q info: {weight_q.to_local().shape}")
        # Output: [2048, 4096] - each GPU has half the rows, all columns
        
        logger.info(f"[Rank {rank}] Placements q: {weight_q.placements}")
        # Output: (Shard(dim=0),) - confirms weight is sharded on dimension 0 (rows)
    
       # INSPECTING WEIGHTS AND OUTPUTS:
        weight_o = self.o.weight  # This is the Linear module's weight (DTensor)
        
        # q.to_local() shows the LOCAL data this GPU holds
        logger.info(f"[Rank {rank}] Full local tensor o info: {weight_o.to_local().shape}")
        # Output: [32, 512, 2048] - each GPU has half the hidden_dim
        
        # weight.to_local() shows the LOCAL weight shard this GPU holds  
        logger.info(f"[Rank {rank}] Full local weight o info: {weight_o.to_local().shape}")
        # Output: [2048, 4096] - each GPU has half the rows, all columns
        
        logger.info(f"[Rank {rank}] Placements o: {weight_o.placements}")
        # Output: (Shard(dim=0),) - confirms weight is sharded on dimension 0 (rows)

        logger.info("after mapping")
        
        # DTensor logger.info shows GLOBAL/LOGICAL view
        logger.info(f"[Rank {rank}] q shape: {q.shape} and type : {q.type()} on device {q.device}")
        # Output: shape [32, 512, 4096] - DTensor shows logical shape!
        
        logger.info(f"[Rank {rank}] k shape: {k.shape} and type : {k.type()} on device {k.device}")
        logger.info(f"[Rank {rank}] v shape: {v.shape} and type : {v.type()} on device {v.device}")
        
        # RESHAPING FOR MULTI-HEAD ATTENTION:
        # With use_local_output=False, DTensor handles distributed reshaping correctly
        # Each head needs hidden_dim/nb_heads = 4096/8 = 512 features
        # DTensor ensures each head gets the right features even though data is sharded
        
        q_trans = q.view(b, L, self.nb_heads, -1).transpose(1, 2)  # [32, 8, 512, 512]
        k_trans = k.view(b, L, self.nb_heads, -1).transpose(1, 2)  # [32, 8, 512, 512]
        v_trans = v.view(b, L, self.nb_heads, -1).transpose(1, 2)  # [32, 8, 512, 512]
        
        logger.info("after reshape")
        logger.info(f"[Rank {rank}] q_trans shape: {q_trans.shape} and type : {q_trans.type()} on device {q_trans.device}")
        # Output: [32, 8, 512, 512] - correct head_dim of 512!
        # Without use_local_output=False, this would be [32, 8, 512, 256] (wrong!)
        
        logger.info(f"[Rank {rank}] k_trans shape: {k_trans.shape} and type : {k_trans.type()} on device {k_trans.device}")
        logger.info(f"[Rank {rank}] v_trans shape: {v_trans.shape} and type : {v_trans.type()} on device {v_trans.device}")

        # ATTENTION COMPUTATION:
        att_scores = q_trans @ k_trans.transpose(-2, -1) / math.sqrt(self.head_dim)  # [32, 8, 512, 512]
        att_scores = F.softmax(att_scores, -1)

        # Creating mask using DTensor
        _full = torch.distributed.tensor.full(
            (L, L), 
            -float("inf"),
            device_mesh=self.device_mesh,
            placements=[Replicate()]  # Mask is replicated across all GPUs
        )
        logger.info(f"mask dist init : {_full.type()} on device {v_trans.device}")
        _mask = torch.triu(_full, diagonal=1)
        att_scores = att_scores + _mask
        
        out = att_scores @ v_trans  # [32, 8, 512, 512]
        out = out.transpose(1,2).reshape(b, L, -1)  # [32, 512, 4096]
        logger.info(f"out shape: {out.shape} and type : {out.type()} on device {out.device}")
        
        # FINAL PROJECTION WITH RowwiseParallel:
        # - self.o weight is sharded differently with RowwiseParallel
        # - GPU 0 has: weight[:, :2048] → shape [4096, 2048]
        # - GPU 1 has: weight[:, 2048:] → shape [4096, 2048]
        # - This is Shard(dim=1) - sharding dimension 1 (columns)
        # - Each GPU computes partial sums, then allreduce
        return self.o(out)


if __name__ == "__main__":
    init_logger()
    _world_size = int(os.environ["WORLD_SIZE"])
    device_type = torch.accelerator.current_accelerator().type
    device_mesh = init_device_mesh(device_type=device_type, mesh_shape=(_world_size,))

    h_dim = 2 ** 12  # 4096

    # Create model with regular nn.Linear layers
    mha = MHA(hidden_dim=h_dim, nb_heads=8, device_mesh=device_mesh)

    # PARALLELIZATION PLAN:
    # - q, k, v use ColwiseParallel: 
    #   * Weight sharded on dim=0 (rows): [2048, 4096] per GPU
    #   * Each GPU computes **different output columns**
    #   * use_local_output=False returns DTensor for correct reshaping
    #
    # - o uses RowwiseParallel:
    #   * Weight sharded on dim=1 (columns): [4096, 2048] per GPU  
    #   * Each GPU computes partial sums
    #   * Output is allreduced to get final result
    #
    # This pairing (Colwise → Rowwise) minimizes communication:
    # - ColwiseParallel output is Shard(dim=-1)
    # - RowwiseParallel expects input Shard(dim=-1)
    # - No redistribution needed between them!
    
    tp_model = parallelize_module(
        module=mha,
        device_mesh=device_mesh,
        parallelize_plan={
            "q": ColwiseParallel(use_local_output=False),  # Weight: [2048, 4096], Output: DTensor
            "k": ColwiseParallel(use_local_output=False),  # Weight: [2048, 4096], Output: DTensor
            "v": ColwiseParallel(use_local_output=False),  # Weight: [2048, 4096], Output: DTensor
            "o": RowwiseParallel(use_local_output=False),  # Weight: [4096, 2048], Output: DTensor
        }
    )

    # After parallelize_module:
    # - All weights are now DTensors with appropriate sharding
    # - Forward/backward passes handle communication automatically
    # - Gradients are properly synchronized across GPUs

    lr = 0.25
    optimizer = torch.optim.AdamW(tp_model.parameters(), lr=lr, foreach=True)

    logger.info(tp_model)
    
    num_iter = 10
    for i in range(num_iter):
        torch.manual_seed(i)
        x = torch.rand(32, 512, h_dim)  # Input: [32, 512, 4096]
        
        # FORWARD PASS FLOW:
        # 1. Input x gets replicated to all GPUs
        # 2. q/k/v projections: each GPU computes its output columns
        # 3. Attention: computed with sharded q/k/v (DTensor handles this)
        # 4. Output projection (o): partial sums computed and allreduced
        # 5. Final output is a DTensor with correct shape
        
        output = tp_model(x)  # Note: calling tp_model, not mha!
        output.sum().backward()
        optimizer.step()

# KEY TAKEAWAYS:
# 1. ColwiseParallel shards weight ROWS but computes output COLUMNS in parallel
#    - Weight per GPU: [out_features//world_size, in_features] = [2048, 4096]
#    - Name comes from parallel computation of output columns
#
# 2. RowwiseParallel shards weight COLUMNS but computes output ROWS (with reduction)
#    - Weight per GPU: [out_features, in_features//world_size] = [4096, 2048]
#    - Name comes from parallel computation contributing to all output rows
#
# 3. use_local_output=False is crucial for attention layers:
#    - Returns DTensor that maintains global view for reshaping
#    - Allows correct multi-head splitting (each head gets full features)
#
# 4. The pairing of ColwiseParallel → RowwiseParallel is efficient:
#    - Output of Colwise is Shard(dim=-1), input of Rowwise expects Shard(dim=-1)
#    - Minimizes redistribution/communication between layers