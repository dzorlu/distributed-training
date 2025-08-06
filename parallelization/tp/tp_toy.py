


import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os

from torch.distributed._tensor.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import (
    parallelize_module,
    ColwiseParallel,
    RowwiseParallel,
)


class MHA(nn.Module):
    def __init__(self, hidden_dim, nb_heads):
        super().__init__()
        self.q = nn.Linear(hidden_dim, hidden_dim)
        self.k = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, hidden_dim)
        
        self.nb_heads = nb_heads
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // nb_heads
        self.o = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        """
        inputs:
            x: [batch, L, hidden_dim]
        """
        b, L = int(x.shape[0]), int(x.shape[1])

        q = self.q(x) # [b, L, hidden_dim]
        v = self.v(x) # [b, L, hidden_dim]
        k = self.k(x) # [b, L, hidden_dim]

        q_trans = q.view(b, L, self.nb_heads, -1).transpose(1, 2) # [b, n_heads, L, head_dim]
        k_trans = k.view(b, L, self.nb_heads, -1).transpose(1, 2) # [b, n_heads, L, head_dim]
        v_trans = v.view(b, L, self.nb_heads, -1).transpose(1, 2) # [b, n_heads, L, head_dim]

        # print(q_trans.shape)
        # print(k_trans.transpose(-2, -1).shape)

        att_scores = q_trans @ k_trans.transpose(-2, -1) / math.sqrt(self.head_dim) # [b, n_heads, L, L]
        att_scores = F.softmax(att_scores, -1)
        _mask = torch.triu(torch.full((L,L), -float("inf")), diagonal=1)
        att_scores += _mask
        out = att_scores @ v_trans # [b, n_heads, L, head_dim]
        print(out.shape)
        out = out.transpose(1,2).reshape(b, L, -1) # [b, L, hidden_dim]

        return self.o(out)


if __name__ == "__main__":

    _world_size = int(os.environ["WORLD_SIZE"])
    device_type = torch.accelerator.current_accelerator().type
    device_mesh = init_device_mesh(device_type=device_type, mesh_shape=(_world_size,))