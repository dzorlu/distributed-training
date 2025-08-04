# PyTorch Profiler Decorator

A simple decorator for enabling PyTorch profiler and NSight profiling in distributed training workflows.

## Features

- 🔥 **PyTorch Profiler Integration**: Automatic profiling with configurable scheduling
- 🚀 **NSight Support**: Optional NVIDIA NSight Systems integration
- 🌐 **Multi-GPU Aware**: Handles distributed training with per-rank trace files
- 📊 **TensorBoard Export**: Automatic TensorBoard trace generation
- 🎛️ **Configurable**: Flexible profiling options and scheduling
- 🔄 **Composable**: Works with other decorators like `@ray`

## Basic Usage

```python
from parallelization import profiler

@profiler()  # Uses sensible defaults: with_flops=True, active=3 steps
def train_model(args):
    # Your training code here
    for step in range(100):
        # Training step
        loss = model(batch)
        loss.backward()
        optimizer.step()
        
        # Important: Step the profiler
        step_profiler()
```

## Advanced Usage

### With Distributed Training

```python
from parallelization import ray, profiler

@ray(num_nodes=2, gpus_per_node=4)
@profiler(
    enabled=True,
    output_dir="./profiles", 
    active=10,
    with_flops=True,
    nsight_enabled=True
)
def distributed_train(args):
    # Distributed training code
    pass
```

### Command Line Integration

The FSDP example shows how to integrate profiler with a simple flag:

```bash
# Enable profiling with sensible defaults
python train.py --profile

# Distributed training with profiling
python train.py --num-nodes 2 --profile
```

## Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `enabled` | `True` | Enable/disable profiling |
| `output_dir` | `"./profiler_logs"` | Directory for profiling outputs |
| `skip_first` | `1` | Initial steps to skip (avoids init overhead) |
| `wait` | `1` | Steps to wait before profiling |
| `warmup` | `1` | Warmup steps |
| `active` | `3` | Active profiling steps |
| `repeat` | `1` | Number of profiling cycles |
| `with_stack` | `False` | Include stack traces |
| `with_flops` | `True` | Include FLOP counts (essential for perf analysis) |
| `with_modules` | `False` | Include module hierarchy |
| `nsight_enabled` | `False` | Enable NSight profiling |
| `nsight_output` | `"nsight_profile"` | NSight output prefix |

## Profiling Schedule

The profiler uses PyTorch's built-in scheduling system following [official best practices](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html):

```
[skip_first] → [wait] → [warmup] → [active] → repeat...
```

- **Skip First**: Initial steps to ignore (default: 1, avoids initialization overhead for FLOP counting)
- **Wait**: Steps to skip before starting profiling cycles
- **Warmup**: Steps to warm up (profiler active but results discarded)  
- **Active**: Steps where profiling data is collected
- **Repeat**: How many times to repeat the cycle

This ensures accurate FLOP measurements by avoiding the overhead from model initialization, memory allocation, and CUDA context setup that typically occurs in the first training step.

## FLOP Measurement

The profiler automatically computes floating-point operations (FLOPs) when `with_flops=True` (default). This is essential for:

- **Model Efficiency Analysis**: Compare FLOPs across different model architectures
- **Performance Optimization**: Identify compute-intensive operations  
- **Hardware Utilization**: Calculate FLOP/s and compare against theoretical peak performance
- **Scaling Analysis**: Understand computational complexity as models grow

Example FLOP analysis output:
```
---------------------------------  ------------  ------------  ------------  ------------  
                             Name    Self FLOPS      CPU total    CUDA total   # of Calls
---------------------------------  ------------  ------------  ------------  ------------
                         aten::mm      1.074G       2.631ms       1.234ms           42
                     aten::conv2d      2.456G      31.931ms      12.543ms           20
                aten::batch_norm      0.128G      14.693ms       3.421ms           20
```

**Note**: FLOP counting adds minimal overhead but provides crucial insights for optimization. The `skip_first=1` default ensures initialization costs don't skew your FLOP measurements.

## Output Files

### PyTorch Profiler
- **TensorBoard traces**: `{output_dir}/rank_{rank}/` 
- **Chrome traces**: `{output_dir}/trace_rank_{rank}.json`
- **Console summary**: Top operations by CUDA time

### NSight (when enabled)
- Profile data collected via CUDA profiler API
- Use NSight Systems to capture and analyze

## Integration Notes

### Training Loop Integration

Always call `step_profiler()` at the end of each training step:

```python
from parallelization.profiler.decorator import step_profiler

for batch in dataloader:
    # Forward pass
    loss = model(batch)
    loss.backward()
    optimizer.step()
    
    # IMPORTANT: Step the profiler
    step_profiler()
```

### Multi-GPU Considerations

- Each GPU rank generates separate trace files
- Only rank 0 prints the profiling summary to avoid spam
- TensorBoard traces are organized by worker name

### Memory Usage

Profiling can consume significant memory. For long training runs:
- Use shorter `active` periods
- Reduce `repeat` cycles  
- Disable `with_stack` unless needed

## Viewing Results

### TensorBoard
```bash
tensorboard --logdir ./profiler_logs
```

### Chrome Tracing
Open `trace_rank_0.json` in Chrome at `chrome://tracing`

### NSight Systems
```bash
nsys profile python train.py --nsight
```

## Tips

1. **Start Simple**: Begin with default settings and `with_flops=True`
2. **Profile Early**: Run profiling on small models first to understand overhead
3. **Batch Operations**: Profile multiple steps at once rather than single steps
4. **Focus Areas**: Use stack traces only when debugging specific issues
5. **Distributed**: Profile on single node first, then scale to multi-node 