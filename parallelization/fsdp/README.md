# ZeRO Stages: Communication Primitives in Practice

## 📊 Memory Efficiency & Communication Patterns

| Stage | What's Sharded | Forward Pass | Backward Pass | Optimizer Step | Memory per GPU |
|-------|----------------|--------------|---------------|----------------|----------------|
| **DDP** | Nothing | No communication | `all_reduce(gradients)` | Local update | 100% |
| **ZeRO-1** | Optimizer states only | No communication | `reduce_scatter(gradients)` | Local update + `all_gather(parameters)` | ~75% |
| **ZeRO-2** | Optimizer + gradients | No communication | `reduce_scatter(gradients)` | Local update + `all_gather(parameters)` | ~50% |
| **ZeRO-3** | Parameters + gradients + optimizer | `all_gather(parameters)` | `all_gather(parameters)` + `reduce_scatter(gradients)` | Local update (no all_gather) | ~33% |

## 🔄 ZeRO-1 & ZeRO-2: Optimizer State (+ Gradient) Partitioning

**ZeRO-1**: Only optimizer states sharded - gradients computed but not stored in sharded form  
**ZeRO-2**: Optimizer states + gradients sharded - gradients stored in sharded form for memory savings  
**Communication**: Identical for both (reduce_scatter more applicable for ZeRO-2 where gradients are actually sharded)

Zero stage 1 is free (in the bandwith limitedregime) memory wins.
communication cost is still 2x params (receive one and pass one) for each GPU.



```python
# ZeRO-1/ZeRO-2 Training Step with 3 GPUs
def zero1_zero2_training_step(model, batch, optimizer):
    # === FORWARD PASS ===
    # No communication - each GPU has full parameters
    loss = model(batch)  # Each GPU: complete model, different batch
    
    # === BACKWARD PASS ===
    loss.backward()  # Each GPU: computes full gradients
    
    # === GRADIENT COMMUNICATION ===
    # reduce_scatter: Takes gradients from all GPUs, sums them, distributes chunks
    all_gradients = [param.grad for param in model.parameters()]
    my_gradient_shard = torch.zeros_like(all_gradients[rank])
    
    # Each GPU gets averaged gradients for the parameter it owns optimizer states for
    dist.reduce_scatter(my_gradient_shard, all_gradients, op=SUM)
    my_gradient_shard /= world_size
    
    # === OPTIMIZER STEP ===
    # Each GPU updates only its owned parameter using its gradient shard
    my_param = list(model.parameters())[rank]
    my_param.grad = my_gradient_shard
    optimizer.step_single_param(my_param)
    
    # === PARAMETER COMMUNICATION ===
    # all_gather: Collect updated parameters from all GPUs to reconstruct full model
    # This is REQUIRED because parameters are still replicated in ZeRO-1/2
    updated_params = [torch.zeros_like(p) for p in model.parameters()]
    for i, param in enumerate(model.parameters()):
        dist.all_gather(updated_params, param.data)
        param.data = updated_params[i]  # Everyone gets full updated model
    
    optimizer.zero_grad()

# Memory: ZeRO-1 ~75%, ZeRO-2 ~50% of DDP
# Communication: reduce_scatter(gradients) + all_gather(parameters)
```

## 🔄 ZeRO-3: Full Parameter Sharding

**What's sharded**: Parameters + gradients + optimizer states  
**Key difference**: Parameters stay sharded - no final all_gather needed

```python
# ZeRO-3 Training Step with 3 GPUs
def zero3_training_step(model, batch, optimizer):
    # === FORWARD PASS ===
    # all_gather: Collect parameter shards to reconstruct full weights for computation
    for layer in model.layers:
        dist.all_gather(full_weight_list, my_weight_shard)
        full_weight = torch.cat(full_weight_list, dim=0)
        
        # Compute with full parameters
        layer_output = F.linear(layer_input, full_weight)
        
        # Immediately discard full parameters
        del full_weight  # Back to sharded state
    
    # === BACKWARD PASS ===
    for layer in reversed(model.layers):
        # all_gather: Reconstruct full weights again for backward computation
        dist.all_gather(full_weight_list, my_weight_shard)
        full_weight = torch.cat(full_weight_list, dim=0)
        
        # Compute gradients
        grad_output.backward()
        
        # reduce_scatter: Sum gradients across GPUs, distribute shards back
        dist.reduce_scatter(my_grad_shard, [grad_weight], op=SUM)
        my_grad_shard /= world_size
        
        del full_weight
    
    # === OPTIMIZER STEP ===
    # Each GPU updates only its parameter shard - NO all_gather needed!
    # Parameters stay sharded across GPUs
    my_param_shard -= lr * my_grad_shard
    optimizer.zero_grad()

# Memory: ~33% of DDP (everything sharded)
# Communication: 2x all_gather(parameters) + reduce_scatter(gradients) per layer
# Key: No final all_gather - parameters remain sharded
```

## 💾 Key Insights

1. **reduce_scatter vs all_reduce**: More applicable for ZeRO-2 where gradients are actually sharded in memory, but used in ZeRO-1 for communication efficiency

2. **Activation Memory**: ZeRO/FSDP provides **NO activation memory savings** - activations are a function of the data batch each GPU processes, not model parameters. Since each GPU processes different data, activations cannot be sharded

3. **ZeRO-3 Communication**: Requires **two all_gather operations per layer** - once in forward pass, once in backward pass to reconstruct parameters for computation

4. **Parameter Gathering**:
   - **ZeRO-1/2**: Need final `all_gather(parameters)` because parameters are replicated
   - **ZeRO-3**: No final all_gather - parameters stay sharded, will be gathered again when needed

5. **Memory vs Communication Trade-off**: More sharding = less memory but more frequent parameter reconstruction

