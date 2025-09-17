from re import I
import torch
import torch.nn as nn
from dataclasses import dataclass
import torch.nn.functional as F
import torch.distributed as dist
from torch.distributed.tensor import DTensor
import contextlib


from .args import ModelArgs
from ..logging import logger
from ..utils import describe_group, log_tensor
import wandb

try:
    from grouped_gemm import ops
except ImportError:
    raise RuntimeError(
        "Grouped GEMM is not available. Please run `pip install --no-build-isolation git+https://github.com/fanshiqing/grouped_gemm@main` (takes less than 5 minutes)"
    )

from torch.distributed.tensor import DTensor



from torch.distributed.tensor.debug import visualize_sharding




class Router(nn.Module):
    """
    Determines which tokens are routed to which experts and reorders the token
    data into an expert-centric layout.

    This module performs three main functions:
    1.  **Routing:** Computes scores and selects the top-k experts for each token.
    2.  **Reordering:** Calculates the permutation needed to group all tokens
        destined for the same expert together.
    3.  **Gathering:** Gathers the actual token data according to the new
        expert-centric order, preparing it for dispatch (e.g., via an
        all-to-all communication primitive).

    Args:
        model_args (MoEModelArgs): A dataclass containing the configuration for
            the MoE layer, including hidden_dim, num_experts, and top_k.
    """
    def __init__(self, model_args: ModelArgs):
        super().__init__()
        
        self.model_args = model_args
        self.router = nn.Linear(model_args.dim, model_args.num_experts, bias=False)
        self.device_mesh = model_args.device_mesh
        self.beta = 0.6
        self.score_func = model_args.score_func
        self.aux_loss_coeff = model_args.aux_loss_coeff
        self.aux_loss = None
        self.register_buffer('step', torch.tensor(0))

        # === LOSS-FREE BALANCING: Expert bias initialization ===
        # Register as buffer (persists but not optimized)
        self.register_buffer('expert_bias', torch.zeros(model_args.num_experts))

    def compute_aux_loss(self, 
        expert_scores: torch.Tensor,
        num_tokens_per_expert: torch.Tensor,
        ) -> torch.Tensor:
        T, K = expert_scores.shape[0], self.model_args.top_k

        p_i = expert_scores.mean(axis=0)
        f_i = num_tokens_per_expert / (T * K)
        logger.info(f"{p_i=} {f_i=}")
        aux_loss = (p_i * f_i).sum() * self.aux_loss_coeff
        logger.info(f"{aux_loss=}")
        self.aux_loss = aux_loss



    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Processes an input tensor to select experts and prepare for MoE dispatch.

        Args:
            x (torch.Tensor): The input tensor of shape `(batch_size * seq_len, hidden_dim)`.

        Returns:
            A tuple containing:
            - torch.Tensor: `x_gathered`: The reordered token data for the experts.
              Shape: `(batch_size * seq_len * top_k, hidden_dim)`.
            - torch.Tensor: `num_tokens_per_expert`: The number of tokens assigned to each expert.
              Shape: `(num_experts,)`.
            - torch.Tensor: `scatter_indices`: The indices needed to scatter the expert outputs
              back to their original token positions. Shape: `(batch_size * seq_len * top_k)`.
            - torch.Tensor: `scores_sorted`: The router scores, sorted in the same
              expert-centric order as `x_gathered`. Shape: `(batch_size * seq_len * top_k)`.
        
        Example Data Flow (batch=1, seq_len=4, top_k=2, num_experts=4):
        - Input `x` shape: `(4, h)`
        - `selected_experts_indices`: `[[0, 2], [1, 3], [0, 3], [1, 2]]`
          (Token 0 goes to experts 0 & 2, Token 1 to 1 & 3, etc.)
        - `token_indices_experts_sorted`: `[0, 2, 1, 3, 0, 3, 1, 2]`
          (The reordered plan: Token 0 for E0, Token 2 for E0, Token 1 for E1, etc.)
        - Output `x_gathered`: Contains data `[x[0], x[2], x[1], x[3], x[0], x[3], x[1], x[2]]`
        """
        # [b*s, h] -> [b*s, num_experts]
        expert_scores = self.router(x)


        # By default, sigmoid or softmax is performed in float32 to avoid loss explosion
        # expert score are normalized such that bias can have a real impact on the routing decision
        # Softmax should be used because sigmoid produces nearly uniform scores (~0.5) across all experts, 
        # making the auxiliary loss insensitive to actual routing imbalance, 
        # while softmax creates meaningful probability variations that properly penalize skewed expert selection.
        if self.score_func == "sigmoid":
            expert_scores = torch.sigmoid(expert_scores.to(torch.float32))
        elif self.score_func == "softmax":
            expert_scores = F.softmax(expert_scores.to(torch.float32), dim=1)
        else:
            raise NotImplementedError(f"Unknown score function {self.score_func}")

        # add noise to the scores to avoid the expert bias from dominating the routing decision
        # initial exploration is important
        self.step += 1
        logger.info(f"{self.step=}")
        noise_scale = (
            self.model_args.moe_noise_scale
            * (1 - self.step.float() / self.model_args.moe_warmup_steps)
        ).clamp(min=0.0)
        if noise_scale > 0:
            logger.info(f"{noise_scale=}")
            expert_scores = expert_scores + torch.randn_like(expert_scores) * noise_scale
        # === LOSS-FREE BALANCING: Apply expert bias ===
        # Add bias to scores BEFORE top-K selection
        # This steers tokens away from overloaded experts
        biased_scores = expert_scores + self.expert_bias.to(expert_scores.dtype).unsqueeze(0)
        
        # [b*s, num_experts], [b*s, num_experts]
        # [[0,2], [1,3], ..] token_A-> [0,2] etc
        _, selected_experts_indices = torch.topk(biased_scores, k=self.model_args.top_k, dim=-1)

        # === ADD: Get original scores for the selected experts ===
        # - The softmax weights should come from the original router scores so gradients flow correctly to the router weights,
        #  not influenced by the bias terms
        top_scores = torch.gather(expert_scores, -1, selected_experts_indices)
        # need to normalize for sigmoid so that the sum of the top_scores is 1
        # this is used for the weighted sum of the expert outputs
        top_scores = top_scores / top_scores.sum(dim=-1, keepdim=True)  

        
        # histogram!
        # [num_experts]
        # Example: [2.0, 2.0, 2.0, 2.0] # each expert gets 2.
        num_tokens_per_expert = torch.bincount(
            selected_experts_indices.view(-1),
            minlength=self.model_args.num_experts
        )
        logger.info(f"{num_tokens_per_expert=}")

        self.compute_aux_loss(expert_scores, num_tokens_per_expert)

        with torch.no_grad():
            counts = num_tokens_per_expert.float()
            E = counts.numel()
            # Some Laplace smoothing
            eps = 1.0
            frac = (counts + eps) / (counts.sum() + eps * E)
            target = 1.0 / E
            bias = target - frac  # zero-sum by construction

            # Some EMA smoothing
            self.expert_bias.mul_(self.beta).add_((1 - self.beta) * bias)
            # sync across router-replica group
            g = self.device_mesh.get_group("dp")
            dist.all_reduce(self.expert_bias, op=dist.ReduceOp.SUM, group=g)
            self.expert_bias.div_(dist.get_world_size(g))

            # keep bounded
            self.expert_bias.clamp_(-0.5, 0.5)
            logger.info(f"{self.expert_bias=}")


        # Flattened view before sort: [0, 2, 1, 3, 0, 3, 1, 2]
        # Meaning: [A→E0, A→E2, B→E1, B→E3, C→E0, C→E3, D→E1, D→E2]
        flat_expert_indices = selected_experts_indices.view(-1)

        # 1) After argsort indices: [0, 4, 2, 6, 1, 7, 3, 5]
        # This reorders to: [A→E0, C→E0, B→E1, D→E1, A→E2, D→E2, B→E3, C→E3]
        sorted_indices = torch.argsort(flat_expert_indices, stable=True)

        # 2) Convert sorted indices back to token indices
        # Result: [0, 2, 1, 3, 0, 3, 1, 2]
        # Meaning: [TokenA, TokenC, TokenB, TokenD, TokenA, TokenD, TokenB, TokenC]
        # This is the crucial index tensor that tells the final scatter operation
        # which original token position each expert output corresponds to.
        scatter_indices = sorted_indices // self.model_args.top_k

        # Gather the actual token data from the token indices;
        # now x is replicated and in order
        # [b*s*top_k, h]
        #visualize_sharding(x)
        #visualize_sharding(scatter_indices)
        x_gathered = x[scatter_indices]
        
        # Also reorder the scores to match the new token order.
        scores_sorted = top_scores.view(-1)[sorted_indices]
        
        return x_gathered, num_tokens_per_expert, scatter_indices, scores_sorted

    def init_weights(self, init_std: float):
        with torch.random.fork_rng():
            torch.manual_seed(42)  # Same seed on all ranks
            nn.init.trunc_normal_(self.router.weight, mean=0.0, std=init_std)



class GroupedExpert(nn.Module):
    def __init__(self, model_args: ModelArgs):
        """
        Implements a grouped MLP computation for multiple experts using grouped GEMM operations.
        
        This module efficiently processes tokens assigned to different experts in a single
        batched operation. Each expert has its own set of MLP weights (w1, w2, w3), and
        tokens are processed by their assigned experts using grouped matrix multiplications.
        
        The MLP follows a SwiGLU-style architecture:
        - Two parallel projections: w1 (gate) and w3 (up)
        - SiLU activation on w1's output
        - Element-wise multiplication of activated w1 and w3 outputs
        - Down projection with w2
        
        Args:
            model_args (MoEModelArgs): Configuration containing:
                - num_experts: Number of expert networks
                - hidden_dim: Input/output dimension of the MLP
                - dim: Hidden dimension of the MLP (intermediate size)
        
        Attributes:
            w1 (nn.Parameter): Gate projection weights. Shape: (num_experts, hidden_dim, dim)
            w2 (nn.Parameter): Down projection weights. Shape: (num_experts, dim, hidden_dim)
            w3 (nn.Parameter): Up projection weights. Shape: (num_experts, hidden_dim, dim)
        """
        super().__init__()
        self.model_args = model_args
        self.device_mesh = model_args.device_mesh
        # fine-grained ratio
        self.w1 = nn.Parameter(torch.empty(model_args.num_experts, model_args.dim, model_args.dim // 4))
        self.w2 = nn.Parameter(torch.empty(model_args.num_experts, model_args.dim // 4, model_args.dim))
        self.w3 = nn.Parameter(torch.empty(model_args.num_experts, model_args.dim, model_args.dim // 4))

    def forward(self, x: torch.Tensor, num_tokens_per_expert: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the expert layer.

        Args:
            x (torch.Tensor): Input tensor dispatched to this expert.
            num_tokens_per_expert (torch.Tensor): Number of tokens assigned to each expert.
        """

        col_group_rank = dist.get_rank(self.device_mesh.get_group("ep"))
        row_group_rank = dist.get_rank(self.device_mesh.get_group("dp"))
        logger.info(f"{col_group_rank=} {row_group_rank=}")
        
        # Create a CPU copy for logging and operations that require it
        num_tokens_per_expert_cpu = num_tokens_per_expert.to("cpu").to(torch.int64)
        logger.info(f"{num_tokens_per_expert_cpu=}")

        # Log to wandb if enabled
        if wandb.run is not None:
            # Create a dictionary for logging: {'expert_0': 4096, 'expert_1': 4096, ...}
            for i, count in enumerate(num_tokens_per_expert_cpu):
                i = i + col_group_rank * len(num_tokens_per_expert_cpu)
                token_dist = {f"expert_{row_group_rank}_{i}": count.item()}
                wandb.log({"token_distribution": token_dist}, commit=False)

        # --- Fused GMM Kernel ---
        # The rest of the forward pass remains the same...
        x_bf16 = x.to(torch.bfloat16)

        if not isinstance(self.w1, DTensor):
            raise RuntimeError("w1 is not a DTensor")
        # The weights are DTensors. We need to convert them to local tensors
        # to use them with non-DTensor aware ops like ops.gmm
        w1_bf16 = self.w1.to_local().to(torch.bfloat16)
        w2_bf16 = self.w2.to_local().to(torch.bfloat16)
        w3_bf16 = self.w3.to_local().to(torch.bfloat16)
    
        # MLP layers with activation
        # Expected batch_sizes.size(0) == num_experts
        #logger.info(f"{x_bf16.shape=}, {w1_bf16.shape=} {num_tokens_per_expert_cpu=}")
        x1 = ops.gmm(x_bf16, w1_bf16, num_tokens_per_expert_cpu, trans_b=False)
        x3 = ops.gmm(x_bf16, w3_bf16, num_tokens_per_expert_cpu, trans_b=False)
        h = F.silu(x1) * x3
        out = ops.gmm(h, w2_bf16, num_tokens_per_expert_cpu, trans_b=False)
        #logger.info(f"out shape: {out.shape} and dtype: {out.dtype} and type: {type(out)} and requires_grad: {out.requires_grad}")
        return out.to(x.dtype)

    def init_weights(self, init_std: float):
        nn.init.trunc_normal_(self.w1, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.w2, mean=0.0, std=init_std)
        nn.init.trunc_normal_(self.w3, mean=0.0, std=init_std)


class MoE(nn.Module):
    """
    A complete Mixture of Experts (MoE) layer that composes a Router and
    GroupedExpert modules.
    """
    def __init__(self, model_args: ModelArgs):
        super().__init__()
        self.model_args = model_args
        self.router = Router(model_args)
        self.experts = GroupedExpert(model_args)

    def forward(self, x: torch.Tensor):
        bsz, seq, dim = x.shape
        #logger.info(f"{bsz=}, {seq=}, {dim=}")
        x_flat = x.reshape(-1, dim)

        # 1. Get routing plan, gathered tokens, and scores from the router.
        x_gathered, num_tokens_per_expert, scatter_indices, scores_sorted = self.router(x_flat)
        
        # 2. Pass the gathered tokens and token counts to the experts.
        # shape (bs*slen*top_k, dim)
        routed_output = self.experts(x_gathered, num_tokens_per_expert)

        # 3. Weight the expert outputs by their router scores.
        weighted_routed_output = routed_output * scores_sorted.unsqueeze(1)

        # 4. Scatter the weighted outputs back to their original token positions.
        # # Accumulate values into buckets
        # src = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        # index = torch.tensor([0, 2, 0, 1, 2])  # which bucket for each element
        # dst = torch.zeros(3)  # 3 buckets
        # dst.scatter_add_(0, index, src)
        # # Result: [4.0, 4.0, 7.0]
        # # Bucket 0: 1.0 + 3.0 = 4.0
        # # Bucket 1: 4.0 = 4.0  
        # # Bucket 2: 2.0 + 5.0 = 7.0
        out_flat = torch.zeros_like(x_flat)
        scatter_indices_expanded = scatter_indices.unsqueeze(1).expand_as(weighted_routed_output)
        out_flat.scatter_add_(
            dim=0,
            index=scatter_indices_expanded,
            src=weighted_routed_output,
        )
        
        # 5. Reshape the output back to the original input shape.
        return out_flat.reshape(bsz, seq, dim)

    def init_weights(self, init_std: float):
        self.router.init_weights(init_std=init_std)
        self.experts.init_weights(init_std=init_std)


    @classmethod
    def from_model_args(cls, model_args: ModelArgs) -> "MoE":
        """
        Initialize a MoE model from a ModelArgs object.

        Args:
            model_args (ModelArgs): Model configuration arguments.

        Returns:
            MoE: MoE model.

        """
        return cls(model_args)