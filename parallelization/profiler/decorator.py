import os
import torch
from functools import wraps
from torch.profiler import profile, ProfilerActivity, schedule


def profiler(
    enabled=True,
    output_dir="./profiler_logs",
    skip_first=1,
    wait=1,
    warmup=1, 
    active=3,
    repeat=1,
    with_stack=False,
    with_flops=True,
    with_modules=False,
    nsight_enabled=False,
    nsight_output="nsight_profile"
):
    """
    PyTorch profiler and NSight decorator for training functions.
    
    Args:
        enabled: Whether to enable profiling (default: True)
        output_dir: Directory to save profiling results (default: "./profiler_logs")
        skip_first: Number of initial steps to skip (default: 1, avoids initialization overhead)
        wait: Number of steps to wait before starting profiling (default: 1)
        warmup: Number of warmup steps (default: 1)
        active: Number of active profiling steps (default: 3)
        repeat: Number of profiling cycles to repeat (default: 1)
        with_stack: Include stack traces in profiling (default: False)
        with_flops: Include FLOP counts (default: True, essential for performance analysis)
        with_modules: Include module hierarchy (default: False)
        nsight_enabled: Enable NSight profiling (default: False)
        nsight_output: NSight output file prefix (default: "nsight_profile")
    
    Usage:
        @profiler(enabled=True, with_flops=True, nsight_enabled=True)
        def main(args):
            # training logic here
            
        # Or combine with ray:
        @ray(num_nodes=2, gpus_per_node=1)
        @profiler(enabled=True, active=5)
        def main(args):
            # distributed training with profiling
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not enabled:
                return func(*args, **kwargs)
            
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            
            # Setup NSight profiling if enabled
            if nsight_enabled:
                try:
                    torch.cuda.cudart().cudaProfilerStart()
                    print(f"🔍 NSight profiling enabled - output: {nsight_output}")
                except Exception as e:
                    print(f"⚠️  NSight profiling failed to start: {e}")
            
            # Configure PyTorch profiler
            activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
            
            profiler_schedule = schedule(
                skip_first=skip_first,
                wait=wait,
                warmup=warmup,
                active=active,
                repeat=repeat
            )
            
            print(f"📊 PyTorch profiler enabled:")
            print(f"   📁 Output directory: {output_dir}")
            print(f"   ⏱️  Schedule - skip_first: {skip_first}, wait: {wait}, warmup: {warmup}, active: {active}, repeat: {repeat}")
            print(f"   🔧 Options - stack: {with_stack}, flops: {with_flops}, modules: {with_modules}")
            if with_flops:
                print(f"   💡 FLOP counting enabled - skipping first {skip_first} step(s) to avoid initialization overhead")
            
            # Determine rank for multi-GPU setups
            rank = int(os.environ.get("LOCAL_RANK", 0))
            world_size = int(os.environ.get("WORLD_SIZE", 1))
            
            # Create rank-specific output filename
            trace_filename = f"trace_rank_{rank}.json"
            if world_size > 1:
                print(f"🌐 Multi-GPU detected - saving rank {rank} trace to {trace_filename}")
            
            with profile(
                activities=activities,
                schedule=profiler_schedule,
                on_trace_ready=torch.profiler.tensorboard_trace_handler(
                    output_dir, 
                    worker_name=f"rank_{rank}"
                ),
                record_shapes=True,
                profile_memory=True,
                with_stack=with_stack,
                with_flops=with_flops,
                with_modules=with_modules
            ) as prof:
                
                # Store profiler in a way that training loop can access it
                # This allows the training loop to call prof.step() appropriately
                if hasattr(torch, '_current_profiler'):
                    original_profiler = torch._current_profiler
                else:
                    original_profiler = None
                
                torch._current_profiler = prof
                
                try:
                    result = func(*args, **kwargs)
                finally:
                    # Restore original profiler state
                    if original_profiler is not None:
                        torch._current_profiler = original_profiler
                    else:
                        if hasattr(torch, '_current_profiler'):
                            delattr(torch, '_current_profiler')
            
            # Export trace to file as well
            trace_path = os.path.join(output_dir, trace_filename)
            prof.export_chrome_trace(trace_path)
            print(f"💾 Chrome trace saved to: {trace_path}")
            
            # Print profiling summary
            if rank == 0:  # Only print summary from rank 0 to avoid spam
                print("\n📈 Profiling Summary:")
                print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
            
            # Stop NSight profiling if enabled
            if nsight_enabled:
                try:
                    torch.cuda.cudart().cudaProfilerStop()
                    print(f"✅ NSight profiling completed")
                except Exception as e:
                    print(f"⚠️  NSight profiling failed to stop: {e}")
            
            return result
            
        return wrapper
    return decorator


def step_profiler():
    """
    Helper function to step the profiler in training loops.
    Call this at the end of each training step.
    """
    if hasattr(torch, '_current_profiler') and torch._current_profiler is not None:
        torch._current_profiler.step() 