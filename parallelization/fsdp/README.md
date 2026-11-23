# ZeRO Stages: Communication Primitives in Practice

## Distilled (2026-11-23)

- **All-reduce = reduce-scatter + all-gather.** Reduce-scatter is accumulation (each hop adds its own copy of the
  chunk). All-gather is just copy. Each costs ~1P per GPU on a ring, so all-reduce is 2P.
- **ZeRO-1**: you do forward and backward. This is fine because all params are on the GPU. Then for the optimizer
  step, each GPU only receives the gradients for the weights whose optimizer state sits on that GPU. Then it updates
  those weights. Then it shares the new params with all GPUs.
- **Why reduce-scatter**: you reduce all gradients across all GPUs, but only send each GPU its slice, rather than
  sending them all.
- **Ring reduce-scatter**: a GPU sends a distinct chunk each step. The next GPU grabs it, adds its own copy of that
  chunk, and passes it along. After N-1 steps each chunk has been summed over all GPUs and lands on the GPU that owns
  it. A GPU also relays partial sums for chunks it does not own; it keeps only its own.
- **All-gather**: same ring, but there is no sum operation, just a copy at each step. After N-1 steps every GPU has
  every slice.
- **ZeRO-2**: each GPU has updated only its slice of params. That needs to be copied over to all, and that is the
  all-gather. ZeRO-2 is free: reduce-scatter + all-gather is exactly DDP's all-reduce, just with the optimizer step in
  between.
- **ZeRO-3**: params stay sharded. All-gather each layer's params in forward, and again in backward, because you need
  the full params to do the backward pass. Then reduce-scatter the gradients.
- **Communication**: DDP, ZeRO-1 and ZeRO-2 are 2P; ZeRO-3 is 3P.

## 📊 Memory Efficiency & Communication Patterns

| Stage | What's Sharded | Forward Pass | Backward Pass | Optimizer Step | Memory per GPU | Bandwidth per Step |
|-------|----------------|--------------|---------------|----------------|----------------|-------------------|
| **DDP** | Nothing | No communication | `all_reduce(gradients)` | Local update | 100% | 2×params |
| **ZeRO-1** | Optimizer states only | No communication | `reduce_scatter(gradients)` | Local update + `all_gather(parameters)` | ~75% | 2×params |
| **ZeRO-2** | Optimizer + gradients | No communication | `reduce_scatter(gradients)` | Local update + `all_gather(parameters)` | ~50% | 2×params |
| **ZeRO-3** | Everything | `all_gather(parameters)` | `all_gather(parameters)` + `reduce_scatter(gradients)` | Local update (params stay sharded) | ~33% | 3×params |

## 📈 Communication Cost Analysis

| Method | Communication Operations | Bandwidth Breakdown | Total Bandwidth |
|--------|-------------------------|-------------------|-----------------|
| **DDP** | `all_reduce(grads)` | 2P | **2×params** |
| **ZeRO-1** | `reduce_scatter(grads)` + `all_gather(params)` | 1P + 1P | **2×params** |
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
    # reduce_scatter: sum gradients across all GPUs, each GPU receives only its slice
    # (the slice matching its optimizer states). Bandwidth: 1×params
    reduce_scatter(gradients)
    
    # === OPTIMIZER STEP ===
    # Each GPU updates ONLY params it has Adam states for
    # GPU 0: updates params[0:N/4] with its momentum_0, variance_0
    # GPU 1: updates params[N/4:N/2] with its momentum_1, variance_1
    
    # === PARAMETER SYNC ===
    # all_gather: Each GPU's updated params → all GPUs
    # Bandwidth: 1×params
    all_gather(parameters)

# Memory: Saves 2×params (Adam states) / world_size
# Bandwidth: 1P (reduce_scatter) + 1P (all_gather) = 2×params
# Same communication as ZeRO-2. The difference is memory: ZeRO-1 keeps the full gradient
# buffer until backward ends; ZeRO-2 reduce-scatters in buckets during backward and frees them.
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

1. **ZeRO-1 and ZeRO-2 match DDP bandwidth.** All use 2×params:
   - DDP: 2P for all_reduce (= reduce_scatter + all_gather on a ring)
   - ZeRO-1/2: 1P for reduce_scatter + 1P for all_gather
   - This makes ZeRO-2 very attractive: 50% memory savings at no extra communication cost

2. **Bandwidth costs** (per training step):
   ```
   DDP:     2P (all_reduce grads)
   ZeRO-1:  2P (reduce_scatter grads + all_gather params)
   ZeRO-2:  2P (reduce_scatter grads + all_gather params)
   ZeRO-3:  3P (2×all_gather params + reduce_scatter grads)
   ```

3. **Communication Frequency**:
   - **ZeRO-1/2**: Once per batch
   - **ZeRO-3**: Per layer (harder to overlap with compute)
