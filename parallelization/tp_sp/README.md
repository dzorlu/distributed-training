# Understanding ColwiseParallel and RowwiseParallel in PyTorch

## Overview

PyTorch's tensor parallel APIs (`ColwiseParallel` and `RowwiseParallel`) implement distributed matrix multiplication by sharding weights and managing activation redistribution through hooks.

## ColwiseParallel

Shards the Linear layer by **columns** (output features), requiring all-gather for full input.

### Implementation Details

```python
class ColwiseParallel:
    def __init__(self, input_layouts=None, output_layouts=None):
        self.input_layouts = input_layouts or Replicate()      
        self.desired_input_layouts = Replicate()  # ALWAYS needs replicated input
        self.output_layouts = output_layouts or Shard(-1)  # Produces feature-sharded
    
    def _partition_linear_fn(self, module, device_mesh):
        # Weight sharding for Linear(512, 2048)
        # Original weight: [2048, 512] (out_features, in_features)
        # Apply Shard(0) → shard along dim 0 (output features)
        # Rank 0: weight[0:1024, :]     → [1024, 512]
        # Rank 1: weight[1024:2048, :]  → [1024, 512]
        for name, param in module.named_parameters():
            dist_param = distribute_tensor(param, device_mesh, [Shard(0)])
    
    def _prepare_input_fn(self, input_layouts, desired_input_layouts, mod, inputs, device_mesh):
        # Pre-forward hook: redistributes input
        input_tensor = inputs[0]
        if not isinstance(input_tensor, DTensor):
            input_tensor = DTensor.from_local(input_tensor, device_mesh, input_layouts)
        
        # Key operation: redistribute to desired layout
        # If input_layouts=Shard(1) and desired=Replicate()
        # This triggers ALL-GATHER internally
        return input_tensor.redistribute(placements=desired_input_layouts)
    
    def _prepare_output_fn(self, output_layouts, use_local_output, mod, outputs, device_mesh):
        # Post-forward hook: ensures output has correct layout
        # Default output is already Shard(-1) from computation
        # Usually no redistribution needed
        if outputs.placements != output_layouts:
            outputs = outputs.redistribute(placements=output_layouts)
        return outputs.to_local() if use_local_output else outputs
```

### Example Forward Pass

```python
# Setup: 2 ranks, Linear(512, 2048)
# Input: [B=4, S=512, H=512] globally, but seq-sharded

# Step 1: Input arrives at each rank
# Rank 0: [4, 256, 512]  # seq[0:256]
# Rank 1: [4, 256, 512]  # seq[256:512]
# Placement: Shard(1)

# Step 2: _prepare_input_fn redistributes
# Shard(1) → Replicate() = ALL-GATHER operation
# Both ranks get: [4, 512, 512]

# Step 3: Local matmul with sharded weights
# Input: [4, 512, 512] @ Weight.T: [512, 1024] = [4, 512, 1024]
# Rank 0 computes features[0:1024]
# Rank 1 computes features[1024:2048]

# Step 4: Output
# Each rank: [4, 512, 1024]
# Placement: Shard(-1) (feature-sharded)
```

## RowwiseParallel

Shards the Linear layer by **rows** (input features), producing partial outputs that need reduction.

### Implementation Details

```python
class RowwiseParallel:
    def __init__(self, input_layouts=None, output_layouts=None):
        self.input_layouts = input_layouts or Shard(-1)
        self.desired_input_layouts = Shard(-1)  # ALWAYS needs feature-sharded input
        self.output_layouts = output_layouts or Replicate()
    
    def _partition_linear_fn(self, module, device_mesh):
        # Weight sharding for Linear(2048, 512)
        # Original weight: [512, 2048] (out_features, in_features)
        # Apply Shard(1) → shard along dim 1 (input features)
        # Rank 0: weight[:, 0:1024]     → [512, 1024]
        # Rank 1: weight[:, 1024:2048]  → [512, 1024]
        for name, param in module.named_parameters():
            if "weight" in name:
                dist_param = distribute_tensor(param, device_mesh, [Shard(1)])  # Note: Shard(1)!
            elif "bias" in name:
                dist_param = distribute_tensor(param, device_mesh, [Replicate()])
    
    def _prepare_input_fn(self, input_layouts, desired_input_layouts, mod, inputs, device_mesh):
        # Usually no-op: input already feature-sharded from ColwiseParallel
        # Shard(-1) → Shard(-1) = NO-OP
        return inputs  # Already correct
    
    def _prepare_output_fn(self, output_layouts, use_local_output, mod, outputs, device_mesh):
        # Post-forward hook: handles reduction and redistribution
        # Default: Replicate()
        # Partial() -> Replicate(). All-reduce
        # After matmul, output has Partial() placement (needs reduction)
        # If output_layouts=Shard(1): Partial() → Shard(1) = REDUCE-SCATTER (SP)
        return outputs.redistribute(placements=output_layouts)
```

### Example Forward Pass

```python
# Setup: 2 ranks, Linear(2048, 512)
# Input from ColwiseParallel: feature-sharded

# Step 1: Input at each rank
# Rank 0: [4, 512, 1024]  # features[0:1024]
# Rank 1: [4, 512, 1024]  # features[1024:2048]
# Placement: Shard(-1)

# Step 2: _prepare_input_fn (no-op)
# Already correct layout

# Step 3: Local matmul with sharded weights
# Input: [4, 512, 1024] @ Weight.T: [1024, 512] = [4, 512, 512]
# Each rank computes partial output (only saw half the input features)
# Placement after matmul: Partial() (implicit)

# Step 4: _prepare_output_fn redistributes
# Partial() → Shard(1) = REDUCE-SCATTER operation
# 1. Sum partials across ranks
# 2. Scatter along sequence dimension
# Rank 0: [4, 256, 512]  # seq[0:256], fully summed
# Rank 1: [4, 256, 512]  # seq[256:512], fully summed
```

## Sequence Parallel Configuration

```python
from torch.distributed.tensor.parallel import parallelize_module, ColwiseParallel, RowwiseParallel
from torch.distributed._tensor import Shard

# Sequence Parallel: maintains sequence sharding between transformer layers
parallelize_plan = {
    # SP → TP transition
    "attention.in_proj": ColwiseParallel(
        input_layouts=Shard(1),      # Expects sequence-sharded input
        # Internally: all-gather → replicated → compute → feature-sharded
    ),
    
    # TP → SP transition  
    "attention.out_proj": RowwiseParallel(
        output_layouts=Shard(1),     # Returns to sequence-sharded
        # Internally: compute with partials → reduce-scatter → seq-sharded
    ),
}

# Apply parallelization
model = parallelize_module(model, device_mesh, parallelize_plan)
```

## Complete Data Flow Example

```python
# 2 ranks, MLP: Linear(512, 2048) → ReLU → Linear(2048, 512)
# Global batch: [B=4, S=512, H=512]

# Initial: Sequence-sharded input
# Rank 0: [4, 256, 512]  # seq[0:256]
# Rank 1: [4, 256, 512]  # seq[256:512]

# === ColwiseParallel(Linear(512, 2048)) ===
# 1. All-gather: [4, 256, 512] → [4, 512, 512]
# 2. Matmul: [4, 512, 512] @ [512, 1024] = [4, 512, 1024]
# Output: [4, 512, 1024] (feature-sharded)

# === ReLU (element-wise, no communication) ===
# [4, 512, 1024] → [4, 512, 1024]

# === RowwiseParallel(Linear(2048, 512)) ===
# 1. Matmul: [4, 512, 1024] @ [1024, 512] = [4, 512, 512] (partial)
# 2. Reduce-scatter: [4, 512, 512] → [4, 256, 512]
# Output: [4, 256, 512] (sequence-sharded, fully reduced)
```

## DTensor Redistribute Operations

The `redistribute()` method automatically determines the collective operation based on placement transitions:

| Source Placement | Target Placement | Collective Operation | Description |
|-----------------|------------------|---------------------|-------------|
| `Shard(dim)` | `Replicate()` | **All-gather** | Collect all shards |
| `Shard(dim1)` | `Shard(dim2)` | **All-to-all** | Reshard different dimension |
| `Replicate()` | `Shard(dim)` | **Scatter** | Split tensor |
| `Partial()` | `Replicate()` | **All-reduce** | Sum and replicate |
| `Partial()` | `Shard(dim)` | **Reduce-scatter** | Sum and shard |

The `Partial()` placement indicates tensors with partial values needing reduction (e.g., after RowwiseParallel's matmul where each rank only saw part of the input features).

## Key Insights

1. **ColwiseParallel** needs full input to compute its subset of outputs → requires all-gather
2. **RowwiseParallel** computes with partial inputs producing partial outputs → requires reduction
3. **Weight sharding differs**: ColwiseParallel uses `Shard(0)`, RowwiseParallel uses `Shard(1)`
4. **Communication efficiency**: Only 2 collectives for entire MLP (all-gather at entry, reduce-scatter at exit)
5. **Sequence parallelism**: Maintains sequence sharding between layers, avoiding replicated activations