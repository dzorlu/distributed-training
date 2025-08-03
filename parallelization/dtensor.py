import os
import torch
import torch.distributed as dist
from torch.distributed import init_device_mesh
from torch.distributed.tensor import DTensor, Shard, Replicate

def setup_distributed():
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    dist.init_process_group("nccl")
    
    return rank, local_rank, world_size, device

def example_1d_fsdp_tensor(rank, device):
    """1D mesh: FSDP-style parameter sharding"""
    print(f"\n=== Rank {rank}: 1D FSDP Tensor Sharding ===")
    
    mesh_1d = init_device_mesh("cuda", mesh_shape=(2,), mesh_dim_names=["fsdp"])
    
    # Create parameter tensor
    local_tensor = torch.randn(1024, 512, device=device)
    dtensor_fsdp = DTensor.from_local(local_tensor, mesh_1d, [Shard(0)])
    
    print(f"  Local tensor: {local_tensor.shape}")
    print(f"  Global DTensor: {dtensor_fsdp.shape}")                    # [2048, 512] - stacked
    print(f"  Local storage: {dtensor_fsdp.to_local().shape}")          # [1024, 512] - each GPU
    print(f"  Memory per GPU: {dtensor_fsdp.to_local().numel() * 4 / 1024:.0f}KB")
    print(f"  Total memory: {dtensor_fsdp.numel() * 4 / 1024:.0f}KB")
    
    return dtensor_fsdp

def example_1d_fsdp_linear(rank, device):
    """1D mesh: FSDP-style linear layer sharding"""
    print(f"\n=== Rank {rank}: 1D FSDP Linear Layer ===")
    
    mesh_1d = init_device_mesh("cuda", mesh_shape=(2,), mesh_dim_names=["fsdp"])
    
    linear = torch.nn.Linear(1024, 512, bias=False, device=device)
    
    # Convert to sharded DTensor (FSDP pattern)
    dtensor_weight = DTensor.from_local(linear.weight, mesh_1d, [Shard(0)])
    linear.weight = torch.nn.Parameter(dtensor_weight)
    
    print(f"  Original weight: [out_features=512, in_features=1024]")
    print(f"  Global weight: {linear.weight.shape}")                    # [1024, 1024] - combined
    print(f"  Local weight: {linear.weight.to_local().shape}")          # [512, 1024] - per GPU
    print(f"  Sharding: GPU0=[0:512, :], GPU1=[512:1024, :]")
    
    # Forward pass test
    x = torch.randn(32, 1024, device=device)
    output = linear(x)
    print(f"  Forward: {x.shape} → {output.shape}")
    print(f"  Computation: Each GPU computes [32,1024] @ [512,1024].T = [32,512]")
    print(f"  Result: All-gather combines outputs → [32,1024]")
    
    return linear

def example_1d_tensor_parallel_colwise(rank, device):
    """1D mesh: Column-wise tensor parallelism"""
    print(f"\n=== Rank {rank}: 1D Column-wise Tensor Parallel ===")
    
    mesh_1d = init_device_mesh("cuda", mesh_shape=(2,), mesh_dim_names=["tp"])
    
    linear = torch.nn.Linear(1024, 2048, bias=False, device=device)
    
    # Column-wise: Split output features (dimension 0)
    dtensor_weight = DTensor.from_local(linear.weight, mesh_1d, [Shard(0)])
    linear.weight = torch.nn.Parameter(dtensor_weight)
    
    print(f"  Weight shape: [out_features=2048, in_features=1024]")
    print(f"  Global weight: {linear.weight.shape}")                    # [2048, 1024]
    print(f"  Local weight: {linear.weight.to_local().shape}")          # [1024, 1024] - per GPU
    print(f"  Sharding: GPU0=out[0:1024], GPU1=out[1024:2048]")
    
    # Forward pass
    x = torch.randn(32, 1024, device=device)
    output = linear(x)
    print(f"  Forward: {x.shape} → {output.shape}")
    print(f"  Computation: Each GPU: [32,1024] @ [1024,1024] = [32,1024]")
    print(f"  Result: All-gather combines → [32,2048]")
    
    return linear

def example_1d_tensor_parallel_rowwise(rank, device):
    """1D mesh: Row-wise tensor parallelism"""
    print(f"\n=== Rank {rank}: 1D Row-wise Tensor Parallel ===")
    
    mesh_1d = init_device_mesh("cuda", mesh_shape=(2,), mesh_dim_names=["tp"])
    
    linear = torch.nn.Linear(2048, 512, bias=False, device=device)
    
    # Row-wise: Split input features (dimension 1)
    dtensor_weight = DTensor.from_local(linear.weight, mesh_1d, [Shard(1)])
    linear.weight = torch.nn.Parameter(dtensor_weight)
    
    print(f"  Weight shape: [out_features=512, in_features=2048]")
    print(f"  Global weight: {linear.weight.shape}")                    # [512, 2048]
    print(f"  Local weight: {linear.weight.to_local().shape}")          # [512, 1024] - per GPU
    print(f"  Sharding: GPU0=in[0:1024], GPU1=in[1024:2048]")
    
    # Forward pass (assumes input is sharded from column-wise layer)
    x = torch.randn(32, 2048, device=device)
    output = linear(x)
    print(f"  Forward: {x.shape} → {output.shape}")
    print(f"  Computation: Each GPU: [32,1024] @ [512,1024].T = [32,512]")
    print(f"  Result: All-reduce sums partials → [32,512]")
    
    return linear

def example_2d_hybrid_parallelism(rank, device):
    """2D mesh: Hybrid data + tensor parallelism"""
    print(f"\n=== Rank {rank}: 2D Hybrid Parallelism ===")
    
    mesh_2d = init_device_mesh("cuda", mesh_shape=(2, 2), mesh_dim_names=["dp", "tp"])
    coord = mesh_2d.get_coordinate()
    
    print(f"  Mesh coordinate: {coord} (dp_group={coord[0]}, tp_group={coord[1]})")
    
    linear = torch.nn.Linear(1024, 512, bias=False, device=device)
    
    # Hybrid: Replicate across DP, Shard across TP
    dtensor_weight = DTensor.from_local(linear.weight, mesh_2d, [Replicate(), Shard(0)])
    linear.weight = torch.nn.Parameter(dtensor_weight)
    
    print(f"  Global weight: {linear.weight.shape}")                    # [512, 1024]
    print(f"  Local weight: {linear.weight.to_local().shape}")          # [256, 1024] - per GPU
    print(f"  DP groups: Same weights across dp_group dimension")
    print(f"  TP groups: Sharded weights across tp_group dimension")
    
    # Forward pass
    x = torch.randn(32, 1024, device=device)
    output = linear(x)
    print(f"  Forward: {x.shape} → {output.shape}")
    print(f"  Each DP group processes different batch data")
    print(f"  Each TP group handles different output features")
    
    return linear

def example_2d_full_tensor_parallel(rank, device):
    """2D mesh: Full tensor parallelism (both dimensions)"""
    print(f"\n=== Rank {rank}: 2D Full Tensor Parallelism ===")
    
    mesh_2d = init_device_mesh("cuda", mesh_shape=(2, 2), mesh_dim_names=["tp_out", "tp_in"])
    coord = mesh_2d.get_coordinate()
    
    print(f"  Mesh coordinate: {coord}")
    
    # Create weight matrix
    local_weight = torch.randn(256, 256, device=device)
    dtensor_weight = DTensor.from_local(local_weight, mesh_2d, [Shard(0), Shard(1)])
    
    print(f"  Local weight: {local_weight.shape}")                      # [256, 256] - per GPU
    print(f"  Global weight: {dtensor_weight.shape}")                   # [512, 512] - combined
    print(f"  Sharding: GPU(i,j) = weight[i*256:(i+1)*256, j*256:(j+1)*256]")
    print(f"  Memory per GPU: {dtensor_weight.to_local().numel() * 4 / 1024:.0f}KB")
    print(f"  Memory reduction: 4x (each GPU has 1/4 of parameters)")
    
    return dtensor_weight

def example_mlp_block_tensor_parallel(rank, device):
    """Complete MLP block with optimal tensor parallelism"""
    print(f"\n=== Rank {rank}: MLP Block Tensor Parallel ===")
    
    mesh_1d = init_device_mesh("cuda", mesh_shape=(2,), mesh_dim_names=["tp"])
    
    # Standard transformer MLP: up_proj → activation → down_proj
    up_proj = torch.nn.Linear(1024, 4096, bias=False, device=device)      # Expand
    down_proj = torch.nn.Linear(4096, 1024, bias=False, device=device)    # Contract
    
    # Up projection: Column-wise (split output features)
    up_proj.weight = torch.nn.Parameter(
        DTensor.from_local(up_proj.weight, mesh_1d, [Shard(0)])
    )
    
    # Down projection: Row-wise (split input features)
    down_proj.weight = torch.nn.Parameter(
        DTensor.from_local(down_proj.weight, mesh_1d, [Shard(1)])
    )
    
    print(f"  Up proj weight: {up_proj.weight.shape} → local: {up_proj.weight.to_local().shape}")
    print(f"  Down proj weight: {down_proj.weight.shape} → local: {down_proj.weight.to_local().shape}")
    print(f"  Pattern: Column-wise → Row-wise (optimal communication)")
    
    # Forward pass
    x = torch.randn(32, 1024, device=device)
    
    # MLP forward
    hidden = up_proj(x)                                                  # [32, 4096] - sharded
    hidden = torch.nn.functional.gelu(hidden)                           # Activation
    output = down_proj(hidden)                                           # [32, 1024] - all-reduced
    
    print(f"  Forward pass:")
    print(f"    Input: {x.shape}")
    print(f"    After up_proj: {hidden.shape} (sharded across GPUs)")
    print(f"    After down_proj: {output.shape} (all-reduced)")
    print(f"    Communication: Zero between layers, one all-reduce at end")
    
    return up_proj, down_proj

def example_attention_tensor_parallel(rank, device):
    """Multi-head attention with tensor parallelism"""
    print(f"\n=== Rank {rank}: Attention Tensor Parallel ===")
    
    mesh_1d = init_device_mesh("cuda", mesh_shape=(2,), mesh_dim_names=["tp"])
    
    hidden_size = 1024
    num_heads = 16
    head_dim = hidden_size // num_heads
    
    # Attention projections
    q_proj = torch.nn.Linear(hidden_size, hidden_size, bias=False, device=device)
    k_proj = torch.nn.Linear(hidden_size, hidden_size, bias=False, device=device)
    v_proj = torch.nn.Linear(hidden_size, hidden_size, bias=False, device=device)
    o_proj = torch.nn.Linear(hidden_size, hidden_size, bias=False, device=device)
    
    # Q, K, V: Column-wise (split attention heads)
    for name, proj in [("Q", q_proj), ("K", k_proj), ("V", v_proj)]:
        proj.weight = torch.nn.Parameter(
            DTensor.from_local(proj.weight, mesh_1d, [Shard(0)])
        )
        print(f"  {name} proj: {proj.weight.shape} → local: {proj.weight.to_local().shape}")
    
    # O: Row-wise (combine attention heads)
    o_proj.weight = torch.nn.Parameter(
        DTensor.from_local(o_proj.weight, mesh_1d, [Shard(1)])
    )
    print(f"  O proj: {o_proj.weight.shape} → local: {o_proj.weight.to_local().shape}")
    
    print(f"  Attention heads: {num_heads} total, {num_heads//2} per GPU")
    print(f"  Head dimension: {head_dim}")
    
    # Forward pass
    batch_size, seq_len = 32, 512
    x = torch.randn(batch_size, seq_len, hidden_size, device=device)
    
    q = q_proj(x)  # [32, 512, 1024] - each GPU handles 8 heads worth
    k = k_proj(x)  # [32, 512, 1024] - each GPU handles 8 heads worth
    v = v_proj(x)  # [32, 512, 1024] - each GPU handles 8 heads worth
    
    # Simplified attention (normally reshape to heads, compute attention, etc.)
    attn_output = v  # Placeholder for actual attention computation
    final_output = o_proj(attn_output)
    
    print(f"  Forward pass:")
    print(f"    Input: {x.shape}")
    print(f"    Q,K,V: {q.shape} (each GPU: subset of heads)")
    print(f"    Output: {final_output.shape}")
    print(f"    Communication: Only all-reduce in O projection")
    
    return q_proj, k_proj, v_proj, o_proj

def main():
    rank, local_rank, world_size, device = setup_distributed()
    
    print(f"Rank {rank}: World size {world_size}, Device {device}")
    
    if world_size >= 2:
        # 1D examples (2 GPUs minimum)
        example_1d_fsdp_tensor(rank, device)
        example_1d_fsdp_linear(rank, device)
        example_1d_tensor_parallel_colwise(rank, device)
        example_1d_tensor_parallel_rowwise(rank, device)
        example_mlp_block_tensor_parallel(rank, device)
        example_attention_tensor_parallel(rank, device)
        
        if world_size >= 4:
            # 2D examples (4 GPUs minimum)
            example_2d_hybrid_parallelism(rank, device)
            example_2d_full_tensor_parallel(rank, device)
    
    dist.destroy_process_group()

if __name__ == "__main__":
    main()