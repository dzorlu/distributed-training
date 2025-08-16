# ZeRO Stages: Communication Primitives in Practice

## 📊 Memory Efficiency & Communication Patterns

| Stage | What's Sharded | Forward Pass | Backward Pass | Optimizer Step | Memory per GPU | Bandwidth per Step |
|-------|----------------|--------------|---------------|----------------|----------------|-------------------|
| **DDP** | Nothing | No communication | `all_reduce(gradients)` | Local update | 100% | 2×params |
| **ZeRO-1** | Optimizer states only | No communication | `all_reduce(gradients)` | Local update + `all_gather(parameters)` | ~75% | 3×params |
| **ZeRO-2** | Optimizer + gradients | No communication | `reduce_scatter(gradients)` | Local update + `all_gather(parameters)` | ~50% | 2×params |
| **ZeRO-3** | Everything | `all_gather(parameters)` | `all_gather(parameters)` + `reduce_scatter(gradients)` | Local update (params stay sharded) | ~33% | 3×params |

## 📈 Communication Cost Analysis

| Method | Communication Operations | Bandwidth Breakdown | Total Bandwidth |
|--------|-------------------------|-------------------|-----------------|
| **DDP** | `all_reduce(grads)` | 2P | **2×params** |
| **ZeRO-1** | `all_reduce(grads)` + `all_gather(params)` | 2P + 1P | **3×params** |
| **ZeRO-2** | `reduce_scatter(grads)` + `all_gather(params)` | 1P + 1P | **2×params** |
| **ZeRO-3** | 2×`all_gather(params)` + `reduce_scatter(grads)` | 2P + 1P | **3×params** |

## 🔄 ZeRO-1: Optimizer State Partitioning

**What's sharded**: Only Adam optimizer states (momentum, variance)  
**What's replicated**: Parameters and gradients  
**Key insight**: Each GPU can only update 1/N params (those it has optimizer states for)

```python
# ZeRO-1 Training Step with Adam
def zero1_training_step(model, batch):
    # === FORWARD + BACKWARD ===
    loss = model(batch)
    loss.backward()
    
    # === GRADIENT SYNC ===
    # all_reduce: Average gradients across all GPUs
    # Bandwidth: 2×params
    all_reduce(gradients)  # Every GPU now has identical full gradients
    
    # === OPTIMIZER STEP ===
    # Each GPU updates ONLY params it has Adam states for
    # GPU 0: updates params[0:N/4] with its momentum_0, variance_0
    # GPU 1: updates params[N/4:N/2] with its momentum_1, variance_1
    
    # === PARAMETER SYNC ===
    # all_gather: Each GPU's updated params → all GPUs
    # Bandwidth: 1×params
    all_gather(parameters)

# Memory: Saves 2×params (Adam states) / world_size
# Bandwidth: 2P (all_reduce) + 1P (all_gather) = 3×params
```

## 🔄 ZeRO-2: Optimizer State + Gradient Sharding

**What's sharded**: Adam states AND gradients  
**What's replicated**: Parameters only  
**Critical insight**: Gradient shards align with optimizer state shards!

```python
# ZeRO-2 Training Step with Adam
def zero2_training_step(model, batch):
    # === FORWARD + BACKWARD ===
    loss = model(batch)
    loss.backward()
    
    # === GRADIENT SHARDING ===
    # reduce_scatter: Sum gradients and shard them
    # Bandwidth: 1×params
    reduce_scatter(gradients)
    # GPU 0 gets grad[0:N/4], which matches its optimizer states[0:N/4]
    # GPU 1 gets grad[N/4:N/2], which matches its optimizer states[N/4:N/2]
    
    # === OPTIMIZER STEP ===
    # NO all_gather needed! Each GPU has matching grads & optimizer states
    # GPU 0: updates params[0:N/4] using grad_shard[0:N/4] + adam_states[0:N/4]
    # GPU 1: updates params[N/4:N/2] using grad_shard[N/4:N/2] + adam_states[N/4:N/2]
    
    # === PARAMETER SYNC ===  
    # all_gather: Each GPU's updated params → all GPUs
    # Bandwidth: 1×params
    all_gather(parameters)

# Memory: Saves 3×params (grads + Adam states) / world_size
# Bandwidth: 1P (reduce_scatter) + 1P (all_gather) = 2×params ✨
```

## 🔄 ZeRO-3: Full Parameter Sharding

**What's sharded**: Everything - params, gradients, optimizer states  
**Key difference**: Parameters STAY sharded between training steps

```python
# ZeRO-3 Training Step with Adam
def zero3_training_step(model, batch):
    # === FORWARD PASS (per layer) ===
    # all_gather: Reconstruct full params from shards
    # Bandwidth: 1×params (across all layers)
    for layer in model:
        all_gather(layer.weight_shards)  # Get full weight
        output = forward(input, full_weight)
        del full_weight  # Free memory immediately
    
    # === BACKWARD PASS (per layer) ===
    for layer in reversed(model):
        # all_gather: Reconstruct full params again
        # Bandwidth: 1×params (across all layers)
        all_gather(layer.weight_shards)
        
        # Compute gradients
        grads = compute_gradients(full_weight, ...)
        
        # reduce_scatter: Shard gradients across GPUs
        # Bandwidth: 1×params (across all layers)
        reduce_scatter(grads)
        # GPU 0 gets grad_shard[0:N/4] matching its param_shard[0:N/4]
        
        del full_weight
    
    # === OPTIMIZER STEP ===
    # Each GPU updates ONLY its parameter shard
    # No communication - params stay sharded!
    # GPU 0: param_shard[0:N/4] -= lr * grad_shard[0:N/4] / sqrt(variance[0:N/4])

# Memory: Saves 4×params (everything) / world_size  
# Bandwidth: 1P + 1P (all_gather×2) + 1P (reduce_scatter) = 3×params
```

## 💾 Key Insights

1. **ZeRO-2 matches DDP bandwidth!** Both use 2×params bandwidth:
   - DDP: 2P for all_reduce
   - ZeRO-2: 1P for reduce_scatter + 1P for all_gather
   - This makes ZeRO-2 very attractive: 50% memory savings at no extra communication cost

2. **Bandwidth costs** (per training step):
   ```
   DDP:     2P (all_reduce grads)
   ZeRO-2:  2P (reduce_scatter grads + all_gather params)
   ZeRO-1:  3P (all_reduce grads + all_gather params)
   ZeRO-3:  3P (2×all_gather params + reduce_scatter grads)
   ```

3. **Communication Frequency**:
   - **ZeRO-1/2**: Once per batch
   - **ZeRO-3**: Per layer (harder to overlap with compute)
