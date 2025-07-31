import torch
import torch.distributed as dist
import os

# ============================================================================
# SETUP: Device Binding for Multi-GPU Training
# ============================================================================
# When using torchrun --nproc_per_node=N, each process gets environment variables:
# - LOCAL_RANK: Which GPU this process should use (0, 1, 2, ...)
# - RANK: Global process ID across all nodes
# - WORLD_SIZE: Total number of processes

local_rank = int(os.environ['LOCAL_RANK'])  # Get which GPU this process owns
torch.cuda.set_device(local_rank)           # Bind this process to specific GPU
                                           # Prevents "Duplicate GPU detected" error!

def example_reduce():
    """
    REDUCE: Aggregate tensors from all processes to ONE destination process
    
    Pattern: Multiple → One
    Real-world use case: 
    - Collecting final evaluation metrics at master for logging
    - Loss aggregation in validation phase
    - NOT used in modern training loops (bottleneck issues)
    
    Communication: O(N) - bottleneck at destination
    """
    # Each rank creates different tensor values
    # rank 0: [1, 1, 1, 1, 1]
    # rank 1: [2, 2, 2, 2, 2] 
    # rank 2: [3, 3, 3, 3, 3]
    tensor = torch.tensor([dist.get_rank() + 1] * 5, dtype=torch.float32).cuda()
    rank = dist.get_rank()
    print(f"Before reduce on rank {rank}: {tensor}")
    
    # SUM all tensors to rank 0 (dst=0)
    # Only rank 0 gets: [1+2+3, 1+2+3, 1+2+3, 1+2+3, 1+2+3] = [6, 6, 6, 6, 6]
    # Other ranks: tensor unchanged
    dist.reduce(tensor, dst=0, op=dist.ReduceOp.SUM)
    print(f"After reduce on rank {rank}: {tensor}")

def example_all_reduce():
    """
    ALL_REDUCE: Aggregate tensors from all processes to ALL processes
    
    Pattern: Multiple → Multiple (everyone gets same result)
    Real-world use case: 
    - **PRIMARY**: DDP gradient synchronization in data-parallel training
    - Each GPU processes different batch, computes different gradients
    - All GPUs need averaged gradients to update their model replicas
    - Used when model is NOT sharded (full model copy per GPU)
    
    Implementation: Uses Ring AllReduce for bandwidth optimality!
    Communication: O(1) per process - no bottleneck
    
    Ring AllReduce steps:
    1. Reduce-Scatter: Each GPU gets partial sum of one chunk
    2. AllGather: Share partial sums so everyone has complete result
    
    Example DDP workflow:
    loss.backward()  # Each GPU: different gradients
    all_reduce(grads)  # Average gradients across GPUs
    optimizer.step()  # Each GPU: update with averaged gradients
    """
    tensor = torch.tensor([dist.get_rank() + 1] * 5, dtype=torch.float32).cuda()
    print(f"Before all_reduce on rank {dist.get_rank()}: {tensor}")
    
    # ALL ranks get: [1+2+3, 1+2+3, 1+2+3, 1+2+3, 1+2+3] = [6, 6, 6, 6, 6]
    # This is THE operation for distributed training gradient synchronization
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    print(f"After all_reduce on rank {dist.get_rank()}: {tensor}")
    
def example_gather():
    """
    GATHER: Collect tensors from all processes to ONE destination
    
    Pattern: Multiple → One (but keeps tensors separate, no reduction)
    Real-world use case:
    - **PRIMARY**: Collecting evaluation metrics/predictions at master
    - Gathering distributed test results for analysis
    - Collecting embeddings from different data shards
    - Logging aggregated training statistics
    - NOT used for large model parameters (memory bottleneck)
    
    Key constraint: All input tensors must have SAME SHAPE
    Communication: O(N) - bottleneck at destination
    
    Example evaluation workflow:
    local_accuracy = evaluate_batch(my_test_batch)
    gather(all_accuracies, local_accuracy, dst=0)  # Collect at rank 0
    if rank == 0: print(f"Overall accuracy: {mean(all_accuracies)}")
    """
    tensor = torch.tensor([dist.get_rank() + 1] * 5, dtype=torch.float32).cuda()
    
    # Only destination (rank 0) needs to prepare storage containers
    if dist.get_rank() == 0:
        # Create list of containers - one per source rank
        # gather_list[i] will store tensor from rank i
        gather_list = [
            torch.zeros(5, dtype=torch.float32).cuda()  # Must match input tensor shape!
            for _ in range(dist.get_world_size())
        ]
    else:
        gather_list = None  # Non-destination ranks don't need storage
    
    print(f"Before gather on rank {dist.get_rank()}: {tensor}")
    
    # Result at rank 0:
    # gather_list[0] = [1, 1, 1, 1, 1]  # from rank 0
    # gather_list[1] = [2, 2, 2, 2, 2]  # from rank 1  
    # gather_list[2] = [3, 3, 3, 3, 3]  # from rank 2
    dist.gather(tensor, gather_list, dst=0)
    
    if dist.get_rank() == 0:
        print(f"After gather on rank 0: {gather_list}")

def example_all_gather():
    """
    ALL_GATHER: Collect tensors from all processes to ALL processes
    
    Pattern: Multiple → Multiple (everyone gets everyone's tensor)
    Real-world use case:
    - **PRIMARY**: FSDP2 parameter reconstruction before forward/backward
    - Before computation: sharded parameters are all-gathered into full tensors
    - Each GPU owns shard of weight tensor (e.g., weight[0:1365, :])
    - All-gather reconstructs full weight tensor for computation
    - After computation: automatically re-sharded to save memory
    
    This is the "AllGather" phase of Ring AllReduce!
    Communication: O(1) per process using ring algorithm
    
    Example FSDP2 workflow:
    # Before: GPU 0 has weight[0:1365, :], GPU 1 has weight[1365:2730, :], etc.
    all_gather(full_weight, my_weight_shard)  # Reconstruct full tensor
    output = F.linear(input, full_weight)     # Compute with full weight
    # After: automatically re-shard to save memory
    """
    tensor = torch.tensor([dist.get_rank() + 1] * 5, dtype=torch.float32).cuda()
    
    # ALL ranks need storage containers (unlike gather)
    gather_list = [
        torch.zeros(5, dtype=torch.float32).cuda()
        for _ in range(dist.get_world_size())
        ]
    print(f"Before all_gather on rank {dist.get_rank()}: {tensor}")
    
    # ALL ranks get the same result:
    # gather_list[0] = [1, 1, 1, 1, 1]  # from rank 0
    # gather_list[1] = [2, 2, 2, 2, 2]  # from rank 1
    # gather_list[2] = [3, 3, 3, 3, 3]  # from rank 2
    dist.all_gather(gather_list, tensor)
    print(f"After all_gather on rank {dist.get_rank()}: {gather_list}")
    
def example_scatter():
    """
    SCATTER: Distribute different pieces from ONE source to all processes
    
    Pattern: One → Multiple (opposite of gather)
    Use case: Distributing work chunks, initializing sharded parameters
    
    Source rank has list of tensors, each goes to different destination
    Communication: O(N) from source
    """
    if dist.get_rank() == 0:
        # Source creates different tensors for each destination
        scatter_list = [
            torch.tensor([i + 1] * 5, dtype=torch.float32).cuda()
            for i in range(dist.get_world_size())
            ]
        # scatter_list[0] = [1, 1, 1, 1, 1] → goes to rank 0
        # scatter_list[1] = [2, 2, 2, 2, 2] → goes to rank 1  
        # scatter_list[2] = [3, 3, 3, 3, 3] → goes to rank 2
        print(f"Rank 0: Tensor to scatter: {scatter_list}")
    else:
        scatter_list = None  # Non-source ranks don't have data to scatter
        
    # All ranks prepare container to receive their piece
    tensor = torch.zeros(5, dtype=torch.float32).cuda()
    print(f"Before scatter on rank {dist.get_rank()}: {tensor}")
    
    # Each rank receives its designated tensor from scatter_list
    dist.scatter(tensor, scatter_list, src=0)
    print(f"After scatter on rank {dist.get_rank()}: {tensor}")

def example_reduce_scatter():
    """
    REDUCE_SCATTER: Combines reduction + scatter in one operation
    
    Pattern: Multiple lists → Each rank gets one reduced chunk
    Real-world use case:
    - **PRIMARY**: FSDP2 gradient aggregation after backward pass
    - After all-gather for forward/backward, each GPU has full gradients
    - Reduce-scatter: sum gradients + re-shard them for memory efficiency
    - Each GPU gets aggregated gradients only for its owned parameter shard
    - Enables memory-efficient optimizer updates with sharded parameters
    
    This is the "Reduce-Scatter" phase of Ring AllReduce!
    
    Key insight: Each rank provides a LIST of tensors (length = world_size)
    - Position i in each list gets summed and sent to rank i
    - More memory efficient than all_reduce when you only need part of result
    
    Example FSDP2 workflow:
    # After backward: each GPU has full gradients for current layer
    # GPU 0,1,2: grad_weight[0:4096, 0:4096] (full gradient tensor)
    reduce_scatter(my_grad_shard, [grad_weight], op=SUM)
    # Result: GPU 0 gets grad[0:1365,:], GPU 1 gets grad[1365:2730,:], etc.
    # Update: my_weight_shard -= lr * my_grad_shard (memory efficient!)
    """
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    # Each rank creates a list of tensors - one for each destination rank
    # The tensor at position j will be reduced and sent to rank j
    input_tensor = [
        torch.tensor([(rank + 1) * i for i in range(1, 3)], dtype=torch.float32).cuda()**(j+1) 
        for j in range(world_size)
        ]
    
    # Example with 3 GPUs:
    # rank 0: [tensor([1,2]), tensor([1,4]), tensor([1,8])]
    # rank 1: [tensor([2,4]), tensor([4,16]), tensor([8,64])]  
    # rank 2: [tensor([3,6]), tensor([9,36]), tensor([27,216])]
    #         ↑              ↑               ↑
    #      to rank 0      to rank 1      to rank 2
    
    output_tensor = torch.zeros(2, dtype=torch.float32).cuda()
    print(f"Before ReduceScatter on rank {rank}: {input_tensor}")
    
    # After reduce_scatter:
    # rank 0 gets: sum of position 0 tensors = [1,2] + [2,4] + [3,6] = [6,12]
    # rank 1 gets: sum of position 1 tensors = [1,4] + [4,16] + [9,36] = [14,56]
    # rank 2 gets: sum of position 2 tensors = [1,8] + [8,64] + [27,216] = [36,288]
    dist.reduce_scatter(output_tensor, input_tensor, op=dist.ReduceOp.SUM)
    print(f"After ReduceScatter on rank {rank}: {output_tensor}")    


# ============================================================================
# EXECUTION: Initialize distributed environment and run examples
# ============================================================================
# init_process_group() reads environment variables set by torchrun:
# - RANK, LOCAL_RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT
# - Establishes communication backend (NCCL for GPU)
# - Enables collective operations
dist.init_process_group()

print("**reduce**")
example_reduce()
print("**all-reduce**")
example_all_reduce()
dist.barrier()  # Synchronization point - wait for all processes
print("***gather**")
example_gather()
dist.barrier()
print("**all-gather**")
example_all_gather()
dist.barrier()
print("***scatter**")
example_scatter()
dist.barrier()
print("***reduce-scatter**")
example_reduce_scatter()

# ============================================================================
# HOW TO RUN: torchrun --nproc_per_node=3 script.py
# ============================================================================
# This spawns 3 processes, each bound to different GPU
# torchrun automatically sets environment variables for coordination
# Each process runs this entire script, but operates on different data