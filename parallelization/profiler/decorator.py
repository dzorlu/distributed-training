import os
import time
import torch
from functools import wraps
from torch.profiler import profile, ProfilerActivity, schedule
import torch.cuda.memory as _cm
from ..logging import logger


def flop_counter(model, enabled=True, step_to_measure=1):
    """
    Simple FLOP counter decorator that wraps your existing logic.
    
    Usage:
        @flop_counter(model, enabled=args.profile, step_to_measure=1)
        def training_step(x, step_num):
            loss = model(x).sum()
            loss.backward()
            return loss
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, step_num=None, **kwargs):
            if enabled and step_num == step_to_measure:
                from torch.utils.flop_counter import FlopCounterMode
                import time
                
                rank = int(os.environ.get("LOCAL_RANK", 0))
                
                with FlopCounterMode(mods=model, display=True, depth=None) as ftdm:
                    t = time.time()
                    result = func(*args, step_num=step_num, **kwargs)
                    t_lapsed = time.time() - t
                    # TODO: (dzorlu) grab the model name from the model
                    #total_flops = sum(ftdm.flop_counts['FSDPTransformer'].values()) 
                    #total_flops = sum(ftdm.flop_counts['Transformer'].values())  
                    total_flops = sum(ftdm.flop_counts['MoE'].values())                
                    tflops = total_flops / t_lapsed / 1e12
                    logger.info(f"rank {rank} step {step_num} total_flops: {total_flops:,} tflops: {tflops:.2f}")
                    return result
            else:
                return func(*args, step_num=step_num, **kwargs)
        return wrapper
    return decorator

def trace_handler(prof):
    # Get rank for distributed setups
    rank = int(os.environ.get("LOCAL_RANK", 0))
    
    # Only rank 0 prints summary to avoid spam
    if rank == 0:
        logger.info("\n📈 Profiling Summary:")
        output = prof.key_averages().table(sort_by="cuda_time_total", row_limit=10)
        logger.info(output)
    
    # Each rank saves its own trace
    trace_path = f"/tmp/trace_rank_{rank}_step_{prof.step_num}.json"
    prof.export_chrome_trace(trace_path)
    logger.info(f"💾 Chrome trace saved to: {trace_path}")

    # ─── export memory timeline as a Perfetto‐readable counter track ───
    snap_path = f"/tmp/memory_snapshot_rank{rank}_step{prof.step_num}.pkl"
    _cm._dump_snapshot(snap_path)
    logger.info(f"💾 Memory timeline (Allocated vs Reserved) saved to: {snap_path}")

    # # Compute total FLOPs
    total_flops = sum(evt.flops for evt in prof.key_averages() if hasattr(evt, "flops"))
    logger.info(f"\n💯 Total FLOPs (counted ops): {rank} {total_flops:,}")


def profiler(
    enabled=True,
    output_dir="./profiler_logs",
    skip_first=1,
    wait=1,
    warmup=1, 
    active=3,
    repeat=1,
    with_stack=True,
    with_flops=True,
    with_modules=True,
    nsight_enabled=True,
    nsight_output="nsight_profile"
):
    """
    PyTorch profiler and NSight decorator for training functions.
    
    This decorator wraps training functions to automatically collect:
    - CPU/GPU execution times for each operation
    - Memory allocation/deallocation patterns
    - FLOP (floating-point operations) counts
    - Optional: Python stack traces and module hierarchy
    - Optional: NSight Systems profiling data
    
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
            # === EARLY EXIT IF PROFILING DISABLED ===
            if not enabled:
                return func(*args, **kwargs)
            
            # === SETUP OUTPUT DIRECTORY ===
            # Create directory to store profiling results
            os.makedirs(output_dir, exist_ok=True)
            
            # === NSIGHT PROFILING SETUP ===
            # NSight is NVIDIA's low-level GPU profiler
            # It captures kernel execution, memory bandwidth, SM utilization
            if nsight_enabled:
                try:
                    # Start CUDA profiler APIs that NSight can capture
                    torch.cuda.cudart().cudaProfilerStart()
                    logger.info(f"🔍 NSight profiling enabled - output: {nsight_output}")
                except Exception as e:
                    logger.warning(f"⚠️  NSight profiling failed to start: {e}")

            # ─── NEW: export memory timeline as a Perfetto‐readable counter track ───
            #CUDA allocator snapshot (_record_memory_history / _dump_snapshot) 
            # will capture everything from when you invoked _record_memory_history() 
            # up through the moment you call _dump_snapshot(), 
            # regardless of the profiler schedule.
            _cm._record_memory_history()

            
            # === PYTORCH PROFILER CONFIGURATION ===
            # Configure what activities to profile
            activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
            
            # Create profiling schedule - controls when profiling happens
            # Schedule: [skip_first] → [wait] → [warmup] → [active] → repeat...
            profiler_schedule = schedule(
                skip_first=skip_first,  # Skip first N steps (avoids init overhead)
                wait=wait,              # Wait N steps between cycles
                warmup=warmup,          # Warmup N steps (profiler active but results discarded)
                active=active,          # Active profiling for N steps (data collected)
                repeat=repeat           # Repeat the cycle N times
            )
            
            # === PROFILING STATUS DISPLAY ===
            logger.info(f"📊 PyTorch profiler enabled:")
            logger.info(f"   📁 Output directory: {output_dir}")
            logger.info(f"   ⏱️  Schedule - skip_first: {skip_first}, wait: {wait}, warmup: {warmup}, active: {active}, repeat: {repeat}")
            logger.info(f"   🔧 Options - stack: {with_stack}, flops: {with_flops}, modules: {with_modules}")
            if with_flops:
                logger.info(f"   💡 FLOP counting enabled - skipping first {skip_first} step(s) to avoid initialization overhead")
            
            # === DISTRIBUTED TRAINING SUPPORT ===
            # Get rank information for multi-GPU setups
            # Each GPU process gets a unique LOCAL_RANK (0, 1, 2, 3...)
            rank = int(os.environ.get("LOCAL_RANK", 0))
            world_size = int(os.environ.get("WORLD_SIZE", 1))
            
            # Create separate trace files for each GPU rank
            trace_filename = f"trace_rank_{rank}_{int(time.time())}.json"
            if world_size > 1:
                logger.info(f"🌐 Multi-GPU detected - saving rank {rank} trace to {trace_filename}")
            
            # === MAIN PROFILING CONTEXT ===
            with profile(
                activities=activities,                    # What to profile (CPU + CUDA)
                schedule=profiler_schedule,               # When to profile (skip/wait/warmup/active)
                on_trace_ready=trace_handler,
                record_shapes=True,                      # Record tensor shapes for each op
                profile_memory=True,                     # Track memory allocations/deallocations
                with_stack=with_stack,                   # Include Python stack traces (expensive)
                with_flops=with_flops,                   # Count floating-point operations
                with_modules=with_modules                # Include module hierarchy info
            ) as prof:
                
                # === PROFILER STATE MANAGEMENT ===
                # Store profiler globally so training loop can call prof.step()
                # This allows the training loop to signal when each iteration completes
                if hasattr(torch, '_current_profiler'):
                    original_profiler = torch._current_profiler
                else:
                    original_profiler = None
                
                # Make profiler accessible to step_profiler() function
                torch._current_profiler = prof
                
                # === RUN THE ACTUAL TRAINING FUNCTION ===
                result = func(*args, **kwargs)
            
            # === POST-PROCESSING: PRINT SUMMARY ===
            # Only rank 0 prints summary to avoid spam in multi-GPU setups
            if rank == 0:  
                logger.info("\n📈 Profiling Summary:")
                # Show top 10 operations sorted by CUDA time
                logger.info(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
            
            # === NSIGHT CLEANUP ===
            # Stop NSight profiling if it was enabled
            if nsight_enabled:
                try:
                    torch.cuda.cudart().cudaProfilerStop()
                    logger.info(f"✅ NSight profiling completed")
                except Exception as e:
                    logger.warning(f"⚠️  NSight profiling failed to stop: {e}")
            
            return result
            
        return wrapper
    return decorator


def step_profiler():
    """
    Helper function to step the profiler in training loops.
    
    This MUST be called at the end of each training step to:
    1. Signal the profiler that one iteration has completed
    2. Allow the profiler to advance through its schedule (wait/warmup/active phases)
    3. Trigger trace collection when in active phase
    
    The profiler schedule only works if prof.step() is called regularly!
    
    Usage in training loop:
        for batch in dataloader:
            # Forward pass
            loss = model(batch)
            loss.backward()
            optimizer.step()
            
            # CRITICAL: Tell profiler this step is done
            step_profiler()
    """
    # Check if profiler is active (set by the decorator above)
    if hasattr(torch, '_current_profiler') and torch._current_profiler is not None:
        # Advance profiler to next step in schedule
        torch._current_profiler.step() 