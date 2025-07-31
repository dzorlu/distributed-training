# Distributed Collectives: Real-World Applications

This guide explains when and how each distributed communication primitive is used in modern machine learning training.

## 🔄 ALL_REDUCE: Data Parallel Training (DDP)

**Primary Use Case**: Gradient synchronization when model is **NOT sharded**

### When to Use
- **Data Parallel Training (DDP)**: Each GPU has full model replica
- Each GPU processes different data batches
- Need to average gradients computed on different data

### Real-World Example
```python
# DDP Workflow: Each GPU has complete model copy
for batch in dataloader:
    # Step 1: Each GPU processes different batch
    loss = model(batch_i)       # GPU i processes batch_i
    loss.backward()             # GPU i computes full gradients
    
    # Step 2: Average gradients across all GPUs (Ring AllReduce)
    dist.all_reduce(gradients, op=SUM)
    gradients /= world_size
    
    # Step 3: Each GPU updates full model with averaged gradients
    optimizer.step()
```

### Communication Pattern
- **Algorithm**: Ring AllReduce (bandwidth-optimal)
- **Cost**: O(model_size) per GPU, but no bottleneck
- **Memory**: Each GPU stores full model + full gradients

---

## 📤 ALL_GATHER: Parameter Reconstruction (FSDP2)

**Primary Use Case**: Reconstructing full tensors from sharded parameters before computation

### When to Use
- **FSDP2/Parameter Sharding**: Parameters distributed across GPUs
- Before forward/backward pass: need complete parameter tensors
- After computation: parameters automatically re-sharded

### Real-World Example
```python
# FSDP2: Parameters sharded across GPUs
class FSDPLinear(nn.Module):
    def forward(self, x):
        # Before: Each GPU owns shard of weight tensor
        # GPU 0: weight[0:1365, :]    (DTensor shard)
        # GPU 1: weight[1365:2730, :] (DTensor shard)
        # GPU 2: weight[2730:4096, :] (DTensor shard)
        
        # FSDP2 automatically all-gathers to reconstruct full weight
        # All GPUs temporarily have: weight[0:4096, 0:4096] (complete)
        
        output = F.linear(x, self.weight)  # Compute with full we