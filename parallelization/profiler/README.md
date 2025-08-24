# Performance Profiling

This directory contains a unified `@performance_monitor` decorator to analyze the performance of PyTorch training steps.

## `@performance_monitor` Decorator

A stateful decorator that wraps a single training step function. It combines the PyTorch profiler, a FLOP counter, and a communication logger into a single, configurable tool.

### Usage

Instantiate the monitor and apply it to your training step function.

```python
# In your main training script
from parallelization.profiler import performance_monitor

monitor = performance_monitor(
    model,
    enabled=args.profile,
    flop_counter_step=5,  # Run FLOP counter on step 5
    comm_logger_step=2,   # Log communication collectives on step 2
    # ... other profiler schedule arguments ...
)

@monitor
def train_step(batch):
    # ... your training logic ...
    return loss
```

---

## Outputs

The decorator generates several outputs when enabled.

### Console Output

-   **Initial Setup**: Prints the profiler schedule and configuration at the start.
-   **FLOP Counter Table**: On the `flop_counter_step`, prints a detailed table of FLOPs by module.
-   **FLOPs Summary**: A summary line with total FLOPs and TFLOPs/s for the measured step.
-   **Profiler Summary**: After the profiling `active` window, Rank 0 prints a table of the top 10 most time-consuming CUDA operations.

### File Output

-   **PyTorch Profiler Trace**: A JSON trace file for each rank, viewable in `chrome://tracing` or Perfetto.
    -   **Path**: `/tmp/trace_rank_{rank}_step_{step}.json`
-   **Memory Snapshot**: A pickle file detailing GPU memory allocations over time. Can be visualized with `torch.memory_viz`.
    -   **Path**: `/tmp/memory_snapshot_rank{rank}_step{step}.pkl`
-   **Communication Trace**: A text file logging distributed communication collectives for Rank 0.
    -   **Path**: `comm_debug_{step}.txt`
-   **Communication JSON**: A JSON dump of communication events.
    -   **Path**: `comm_mode_log.json` (by default)
-   **NSight Profile**: If `nsight_enabled=True`, generates an NVIDIA Nsight Systems report.
    -   **Path**: `<nsight_output>.nsys-rep`

### Weights & Biases (`wandb`)

-   **TFLOPs**: Logs the calculated TFLOPs/s for the rank on the measured step (`tflops_rank_{rank}`).
-   **Profiler Events**: Logs the `cuda_time_total` and `cpu_time_total` for the top 10 profiled operations under the `prof/` namespace.

---

## Analysis & Visualization

### Example Commands

```bash
# Generate an NSight profile
nsys profile --output=training_profile.nsys-rep python -m train --profile --gpus-per-node 4

# Copy results from a remote machine
rsync -Pavz user@host:/path/to/project/training_profile.nsys-rep logs/
rsync -Pavz user@host:/tmp/trace_rank_0_step_6.json logs/
rsync -Pavz user@host:/tmp/memory_snapshot_rank0_step6.pkl logs/
```

### Useful Links

-   **Perfetto UI (for Chrome traces)**: https://ui.perfetto.dev/
-   **PyTorch Memory Visualization**: https://pytorch.org/memory_viz
-   **HuggingFace Memory Blog**: https://huggingface.co/blog/train_memory

