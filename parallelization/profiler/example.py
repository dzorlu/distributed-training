#!/usr/bin/env python3
"""
Example usage of the profiler decorator.

This example shows how to use the profiler decorator for both 
single-node and distributed training scenarios.
"""

import torch
import torch.nn as nn
from parallelization import ray, profiler
from parallelization.profiler.decorator import step_profiler


class SimpleModel(nn.Module):
    def __init__(self, input_size=128, hidden_size=256, output_size=10):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(), 
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x):
        return self.layers(x)


@profiler(
    enabled=True,
    output_dir="./example_profiles",
    wait=2,
    warmup=2, 
    active=5,
    with_flops=True,
    with_modules=True
)
def single_gpu_example():
    """Example of profiling single GPU training"""
    print("🖥️  Single GPU Profiling Example")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Simulate training
    for step in range(15):  # More than wait+warmup+active to see full cycle
        # Generate random batch
        batch_size = 32
        x = torch.randn(batch_size, 128, device=device)
        y = torch.randint(0, 10, (batch_size,), device=device)
        
        # Training step
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        
        print(f"Step {step}: Loss = {loss.item():.4f}")
        
        # IMPORTANT: Step the profiler
        step_profiler()
    
    print("✅ Single GPU training completed")


@ray(num_nodes=1, gpus_per_node=2)  # Simulate 2 GPUs
@profiler(
    enabled=True,
    output_dir="./distributed_profiles",
    active=3,
    with_flops=True,
    nsight_enabled=False  # Set to True if you have NSight installed
)
def distributed_example():
    """Example of profiling distributed training"""
    print("🌐 Distributed Profiling Example")
    
    # In real distributed training, this would be set by the distributed launcher
    import os
    rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    
    model = SimpleModel().to(device)
    
    # In real FSDP/DDP, you'd wrap the model here
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    print(f"Rank {rank}: Starting training on {device}")
    
    # Simulate training
    for step in range(10):
        batch_size = 16
        x = torch.randn(batch_size, 128, device=device)
        y = torch.randint(0, 10, (batch_size,), device=device)
        
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        
        print(f"Rank {rank}, Step {step}: Loss = {loss.item():.4f}")
        
        # IMPORTANT: Step the profiler
        step_profiler()
    
    print(f"✅ Rank {rank}: Training completed")


def manual_profiler_example():
    """Example of manual profiler usage without decorator"""
    print("🔧 Manual Profiler Example")
    
    from torch.profiler import profile, ProfilerActivity
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleModel().to(device)
    
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        with_flops=True
    ) as prof:
        
        for step in range(5):
            x = torch.randn(32, 128, device=device)
            y = model(x)
            loss = y.sum()
            loss.backward()
            
            prof.step()  # Manual stepping
    
    # Print results
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=5))
    
    # Export trace
    prof.export_chrome_trace("manual_trace.json")
    print("💾 Manual trace saved to manual_trace.json")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Profiler Examples")
    parser.add_argument("--example", choices=["single", "distributed", "manual"], 
                       default="single", help="Which example to run")
    args = parser.parse_args()
    
    if args.example == "single":
        single_gpu_example()
    elif args.example == "distributed": 
        distributed_example()
    elif args.example == "manual":
        manual_profiler_example()
    
    print("\n📊 Check the output directories for profiling results!")
    print("   - TensorBoard: tensorboard --logdir <output_dir>")
    print("   - Chrome: Open .json files in chrome://tracing") 