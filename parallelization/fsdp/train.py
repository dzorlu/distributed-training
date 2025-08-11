import argparse
import os
import time

import torch
from .checkpoint import Checkpointer
from .model import ModelArgs, Transformer
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from .utils import inspect_mixed_precision, inspect_model


# Import the ray and profiler decorators
from parallelization import ray_distributed, profiler, flop_counter
from parallelization.profiler.decorator import step_profiler



def set_modules_to_forward_prefetch(model, num_to_forward_prefetch):
    for i, layer in enumerate(model.layers):
        if i >= len(model.layers) - num_to_forward_prefetch:
            break
        layers_to_prefetch = [
            model.layers[i + j] for j in range(1, num_to_forward_prefetch + 1)
        ]
        layer.set_modules_to_forward_prefetch(layers_to_prefetch)


def set_modules_to_backward_prefetch(model, num_to_backward_prefetch):
    for i, layer in enumerate(model.layers):
        if i < num_to_backward_prefetch:
            continue
        layers_to_prefetch = [
            model.layers[i - j] for j in range(1, num_to_backward_prefetch + 1)
        ]
        layer.set_modules_to_backward_prefetch(layers_to_prefetch)


def main(args):
    rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    torch.distributed.init_process_group(backend="nccl", device_id=device)
    torch.manual_seed(0)
    vocab_size = 1024
    batch_size = 32
    seq_len = 64
    model_args = ModelArgs(
        n_layers=12,
        n_heads=8, 
        vocab_size=vocab_size,
        max_seq_len=seq_len,
        dropout_p=0,
        dim=4096,
    )
    with torch.device("meta"):
        model = Transformer(model_args)
    fsdp_kwargs = {}
    if args.mixed_precision:
        fsdp_kwargs["mp_policy"] = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
        )
    for layer in model.layers:
        fully_shard(layer, **fsdp_kwargs)
    fully_shard(model, **fsdp_kwargs)

    # Log model parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    param_size_mb = total_params * 4 / (1024 * 1024)  # Assuming float32, 4 bytes per param
    
    print(f"📊 Model Statistics:")
    print(f"   🔢 Total parameters: {total_params:,}")
    print(f"   🎯 Trainable parameters: {trainable_params:,}")
    print(f"   💾 Model size: {param_size_mb:.2f} MB")
    print(f"   🧮 Model config: dim={model_args.dim}, layers={model_args.n_layers}, heads={model_args.n_heads}")

    inspect_model(model)

    if args.explicit_prefetching:
        set_modules_to_forward_prefetch(model, num_to_forward_prefetch=2)
        set_modules_to_backward_prefetch(model, num_to_backward_prefetch=2)

    checkpointer = Checkpointer("checkpoints", dcp_api=args.dcp_api)
    if checkpointer.last_training_time is None:
        model.to_empty(device="cuda")
        model.reset_parameters()
    else:
        checkpointer.load_model(model)
    
    if args.mixed_precision:
        inspect_mixed_precision(model)

    optim = torch.optim.Adam(model.parameters(), lr=1e-2)
    if checkpointer.last_training_time is not None:
        checkpointer.load_optim(model, optim)


    @flop_counter(model, enabled=args.profile, step_to_measure=1)
    def training_step(x, step_num=None):
        loss = model(x).sum()
        loss.backward()
        return loss


    for i in range(100):
        if args.explicit_prefetching:
            model.unshard()
        x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        loss = training_step(x, step_num=i)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optim.step()
        optim.zero_grad()
        
        # Step the profiler if profiling is enabled
        step_profiler()

    checkpointer.save(model, optim)
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch FSDP2 example")
    parser.add_argument("--explicit-prefetching", action="store_true", default=False)
    parser.add_argument("--mixed-precision", action="store_true", default=False)
    parser.add_argument("--dcp-api", action="store_true", default=False)
    parser.add_argument("--num-nodes", type=int, default=1, help="Number of nodes for distributed training")
    parser.add_argument("--gpus-per-node", type=int, default=1, help="Number of GPUs per node")
    
    # Profiler arguments
    parser.add_argument("--profile", action="store_true", default=False, help="Enable PyTorch profiler")
    args = parser.parse_args()
    
    # Apply decorators dynamically based on CLI args
    training_func = main
    
    # Apply profiler decorator if requested
    if args.profile:
        training_func = profiler(enabled=True)(training_func)
    
    # Apply ray decorator for distributed training
    distributed_main = ray_distributed(num_nodes=args.num_nodes, gpus_per_node=args.gpus_per_node)(training_func)
    distributed_main(args)