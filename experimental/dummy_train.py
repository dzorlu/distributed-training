import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, distribute_module, distribute_tensor, Replicate
import os

class SimpleModel(nn.Module):
    def __init__(self, dim=8):
        super().__init__()
        self.linear = nn.Linear(dim, dim, bias=False)
    
    def forward(self, x, use_local=False):
        if use_local and isinstance(self.linear.weight, DTensor):
            # Break DTensor tracking
            weight = self.linear.weight.to_local()
            return x @ weight.T
        else:
            # Maintain DTensor tracking
            return self.linear(x)

def test_dtensor_sync():
    # Initialize distributed
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    
    # Create 1D mesh
    device_mesh = init_device_mesh("cuda", (dist.get_world_size(),))
    rank = dist.get_rank()
    
    print(f"\n{'='*50}")
    print(f"Rank {rank}: Starting DTensor sync test")
    print(f"{'='*50}\n")
    
    # Test 1: Maintain DTensor throughout (gradients should sync)
    print(f"[TEST 1] DTensor maintained through forward pass")
    model1 = SimpleModel().cuda()
    
    # Initialize with same weights using seed
    torch.manual_seed(42)
    nn.init.uniform_(model1.linear.weight, -0.1, 0.1)
    
    # Convert to replicated DTensor
    distribute_module(model1, device_mesh, partition_fn=None)
    
    # Create input (different on each rank to simulate real training)
    torch.manual_seed(rank)  # Different input per rank
    x = torch.randn(4, 8, device='cuda')
    # Convert input to DTensor
    x = distribute_tensor(x, device_mesh, [Replicate()])
    
    # Forward and backward
    out = model1(x, use_local=False)
    loss = out.sum()
    loss.backward()
    
    # Check gradient before optimizer step
    grad_sample = model1.linear.weight.grad.flatten()[:3]
    print(f"Rank {rank} - Gradient sample: {grad_sample}")
    
    # Optimizer step
    with torch.no_grad():
        model1.linear.weight -= 0.1 * model1.linear.weight.grad
    
    # Check weight after update
    weight_sample = model1.linear.weight.flatten()[:3]
    print(f"Rank {rank} - Weight after update: {weight_sample}")
    
    dist.barrier()
    print()
    
    ################################################################
    # Test 2: Break DTensor with to_local (gradients won't sync)
    print(f"[TEST 2] DTensor broken with .to_local()")
    model2 = SimpleModel().cuda()
    
    # Initialize with same weights
    torch.manual_seed(42)
    nn.init.uniform_(model2.linear.weight, -0.1, 0.1)
    
    # Convert to replicated DTensor
    distribute_module(model2, device_mesh, partition_fn=None)
    
    # Same different inputs per rank
    torch.manual_seed(rank)
    x = torch.randn(4, 8, device='cuda')
    # Keep x as regular tensor for this test
    
    # Forward with to_local (breaks DTensor tracking)
    out = model2(x, use_local=True)
    loss = out.sum()
    loss.backward()
    
    # Check gradient
    grad_sample = model2.linear.weight.grad.flatten()[:3]
    print(f"Rank {rank} - Gradient sample: {grad_sample}")
    
    # Optimizer step
    with torch.no_grad():
        model2.linear.weight -= 0.1 * model2.linear.weight.grad
    
    # Check weight after update
    weight_sample = model2.linear.weight.flatten()[:3]
    print(f"Rank {rank} - Weight after update: {weight_sample}")
    
    dist.barrier()
    print()
    
    # Test 3: Verify synchronization with allreduce
    print(f"[TEST 3] Manual verification with allreduce")
    test_tensor = model1.linear.weight.flatten()[:3].clone()
    dist.all_reduce(test_tensor, op=dist.ReduceOp.AVG)
    print(f"Rank {rank} - Model1 weight after allreduce: {test_tensor}")
    
    test_tensor2 = model2.linear.weight.flatten()[:3].clone()
    dist.all_reduce(test_tensor2, op=dist.ReduceOp.AVG)
    print(f"Rank {rank} - Model2 weight after allreduce: {test_tensor2}")
    
    print(f"\n{'='*50}")
    print(f"Rank {rank}: Test complete")
    print(f"Expected: Model1 weights identical across ranks (DTensor sync)")
    print(f"Expected: Model2 weights different across ranks (no sync)")
    print(f"{'='*50}\n")
    
    dist.destroy_process_group()

if __name__ == "__main__":
    test_dtensor_sync()