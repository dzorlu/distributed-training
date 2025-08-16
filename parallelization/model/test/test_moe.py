# parallelization/model/train.py
import argparse
import torch
from parallelization.model.moe import MoE, MoEModelArgs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--gpus-per-node', type=int, default=1)
    args = parser.parse_args()
    
    # Setup
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(0)
    
    # Create model
    moe_args = MoEModelArgs(hidden_dim=16, num_experts=4, top_k=2, dim=32)
    model = MoE(moe_args).to(device)
    
    # Create input
    x = torch.randn(2, 3, 16).to(device)
    
    # Forward pass
    if args.profile:
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, 
                       torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True
        ) as prof:
            y = model(x)
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    else:
        y = model(x)
    
    # Verify
    assert y.shape == x.shape
    assert torch.isfinite(y).all()
    print(f"✓ Forward pass successful on {device}")
    print(f"  Input shape: {x.shape}, Output shape: {y.shape}")


if __name__ == '__main__':
    main()