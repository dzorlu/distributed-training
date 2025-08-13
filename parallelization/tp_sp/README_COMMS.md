# Tensor Parallelism (TP) and Sequence Parallelism (SP) Communication Patterns

## Standard Tensor Parallelism (TP Only)

### Attention Block Communication
```
+---------------+------------------------+---------------------------+-----------------------------+----------------+
| Component     | Weight Sharding        | Forward Pass              | Backward Pass               | Memory per GPU |
+---------------+------------------------+---------------------------+-----------------------------+----------------+
| Q projection  | ColwiseParallel        | No communication          | (contributes to combined    | 1/N weights    |
|               | W[:h/N,:] per GPU      | (b,s,h) → (b,s,h/N)       | input gradient below)       |                |
+---------------+------------------------+---------------------------+-----------------------------+----------------+
| K projection  | ColwiseParallel        | No communication          | (contributes to combined    | 1/N weights    |
|               | W[:h/N,:] per GPU      | (b,s,h) → (b,s,h/N)       | input gradient below)       |                |
+---------------+------------------------+---------------------------+-----------------------------+----------------+
| V projection  | ColwiseParallel        | No communication          | (contributes to combined    | 1/N weights    |
|               | W[:h/N,:] per GPU      | (b,s,h) → (b,s,h/N)       | input gradient below)       |                |
+---------------+------------------------+---------------------------+-----------------------------+----------------+
| Attention ops | No weights             | No communication          | No communication            | -              |
|               |                        | (b,s,h/N) → (b,s,h/N)     | (b,s,h/N) → (b,s,h/N)       |                |
+---------------+------------------------+---------------------------+-----------------------------+----------------+
| O projection  | RowwiseParallel        | all_reduce(output)        | No communication            | 1/N weights    |
|               | W[:,:h/N] per GPU      | y_partial:(b,s,h)         | dX:(b,s,h/N) → (b,s,h/N)    |                |
|               |                        | → y_full:(b,s,h)          |                             |                |
+---------------+------------------------+---------------------------+-----------------------------+----------------+
| Input grad    | -                      | -                         | all_reduce(dx_combined)     | -              |
|               |                        |                           | Combines Q,K,V input grads  |                |
+---------------+------------------------+---------------------------+-----------------------------+----------------+
| TOTAL         | All weights 1/N        | 1 all_reduce              | 1 all_reduce                | 25% (N=4)      |
+---------------+------------------------+---------------------------+-----------------------------+----------------+
```

### MLP Block Communication
```
+---------------+------------------------+---------------------------+-----------------------------+----------------+
| Component     | Weight Sharding        | Forward Pass              | Backward Pass               | Memory per GPU |
+---------------+------------------------+---------------------------+-----------------------------+----------------+
| FC1           | ColwiseParallel        | No communication          | (contributes to             | 1/N weights    |
|               | W[:4h/N,:] per GPU     | (b,s,h) → (b,s,4h/N)      | input gradient below)       |                |
+---------------+------------------------+---------------------------+-----------------------------+----------------+
| GeLU          | No weights             | No communication          | No communication            | -              |
|               |                        | (b,s,4h/N) → (b,s,4h/N)   | (b,s,4h/N) → (b,s,4h/N)     |                |
+---------------+------------------------+---------------------------+-----------------------------+----------------+
| FC2           | RowwiseParallel        | all_reduce(output)        | No communication            | 1/N weights    |
|               | W[:,:4h/N] per GPU     | y_partial:(b,s,h)         | dX:(b,s,4h/N) → (b,s,4h/N)  |                |
|               |                        | → y_full:(b,s,h)          |                             |                |
+---------------+------------------------+---------------------------+-----------------------------+----------------+
| Input grad    | -                      | -                         | all_reduce(dx)              | -              |
|               |                        |                           | Input gradient for MLP      |                |
+---------------+------------------------+---------------------------+-----------------------------+----------------+
| TOTAL         | All weights 1/N        | 1 all_reduce              | 1 all_reduce                | 25% (N=4)      |
+---------------+------------------------+---------------------------+-----------------------------+----------------+
```

### Full Transformer Block with TP
```python
# Forward Pass
def transformer_block_tp_forward(x, weights):
    # === ATTENTION BLOCK ===
    # Q, K, V projections (no communication)
    q_local = x @ W_q_local  # (b,s,h) → (b,s,h/N)
    k_local = x @ W_k_local  # (b,s,h) → (b,s,h/N)
    v_local = x @ W_v_local  # (b,s,h) → (b,s,h/N)
    
    # Attention computation (no communication)
    attn_out_local = attention(q_local, k_local, v_local)  # (b,s,h/N)
    
    # O projection with all_reduce
    o_partial = attn_out_local @ W_o_local  # (b,s,h/N) → (b,s,h)
    attn_output = all_reduce(o_partial)  # ← COMM 1: 2×b×s×h bandwidth
    
    x = x + attn_output  # Residual
    x = layer_norm(x)
    
    # === MLP BLOCK ===
    # FC1 (no communication)
    hidden_local = x @ W_fc1_local  # (b,s,h) → (b,s,4h/N)
    hidden_local = gelu(hidden_local)
    
    # FC2 with all_reduce
    mlp_partial = hidden_local @ W_fc2_local  # (b,s,4h/N) → (b,s,h)
    mlp_output = all_reduce(mlp_partial)  # ← COMM 2: 2×b×s×h bandwidth
    
    output = x + mlp_output  # Residual
    return output

# Total Forward Communication: 2 all_reduces = 4×b×s×h bandwidth
```

## Tensor + Sequence Parallelism (TP+SP)

### Combined Communication Pattern
```
+---------------+----------+------------------------+---------------------------+-----------------------------+----------------+
| Component     | Region   | Weight Sharding        | Forward Pass              | Backward Pass               | Memory per GPU |
+---------------+----------+------------------------+---------------------------+-----------------------------+----------------+
| LayerNorm     | SP       | No sharding            | No communication          | No communication            | Full weights   |
|               |          | (small params)         | Input: (b,s/N,h)          | Operates on (b,s/N,h)       | (negligible)   |
+---------------+----------+------------------------+---------------------------+-----------------------------+----------------+
| SP→TP trans   | -        | -                      | all_gather(activations)   | reduce_scatter(grads)       | -              |
|               |          |                        | (b,s/N,h) → (b,s,h)       | (b,s,h) → (b,s/N,h)         |                |
+---------------+----------+------------------------+---------------------------+-----------------------------+----------------+
| Attention     | TP       | Q,K,V,O: 1/N each      | No communication within   | No communication within     | 1/N weights    |
| Block         |          |                        | attention computation     | attention computation       |                |
+---------------+----------+------------------------+---------------------------+-----------------------------+----------------+
| TP→SP trans   | -        | -                      | reduce_scatter(output)    | all_gather(grads)           | -              |
|               |          |                        | (b,s,h) → (b,s/N,h)       | (b,s/N,h) → (b,s,h)         |                |
+---------------+----------+------------------------+---------------------------+-----------------------------+----------------+
| LayerNorm     | SP       | No sharding            | No communication          | No communication            | Full weights   |
+---------------+----------+------------------------+---------------------------+-----------------------------+----------------+
| SP→TP trans   | -        | -                      | all_gather(activations)   | reduce_scatter(grads)       | -              |
+---------------+----------+------------------------+---------------------------+-----------------------------+----------------+
| MLP Block     | TP       | FC1,FC2: 1/N each      | No communication within   | No communication within     | 1/N weights    |
|               |          |                        | MLP computation           | MLP computation             |                |
+---------------+----------+------------------------+---------------------------+-----------------------------+----------------+
| TP→SP trans   | -        | -                      | reduce_scatter(output)    | all_gather(grads)           | -              |
|               |          |                        | (b,s,h) → (b,s/N,h)       | (b,s/N,h) → (b,s,h)         |                |
+---------------+----------+------------------------+---------------------------+-----------------------------+----------------+
| TOTAL         |          | All weights: 1/N       | 2 all_gather +            | 2 reduce_scatter +          | ~25% weights   |
|               |          |                        | 2 reduce_scatter          | 2 all_gather                | Less activation|
+---------------+----------+------------------------+---------------------------+-----------------------------+----------------+
```

### Full Transformer Block with TP+SP
```python
# Forward Pass with TP+SP
def transformer_block_tp_sp_forward(x_sp, weights):
    # x_sp shape: (b, s/N, h) - sequence dimension is sharded
    
    # === ATTENTION BLOCK ===
    x_sp = layer_norm(x_sp)  # (b,s/N,h) - operates on sharded sequence
    
    # SP→TP transition: gather full sequence for attention
    x_tp = all_gather(x_sp, dim=1)  # (b,s/N,h) → (b,s,h) ← COMM 1: 2×b×s×h bandwidth
    
    # Attention computation in TP region (no communication)
    q_local = x_tp @ W_q_local  # (b,s,h) → (b,s,h/N)
    k_local = x_tp @ W_k_local  # (b,s,h) → (b,s,h/N)
    v_local = x_tp @ W_v_local  # (b,s,h) → (b,s,h/N)
    attn_out_local = attention(q_local, k_local, v_local)
    attn_output = attn_out_local @ W_o_local  # (b,s,h/N) → (b,s,h)
    # Note: NO all_reduce here!
    
    # TP→SP transition: scatter sequence dimension
    attn_sp = reduce_scatter(attn_output, dim=1)  # (b,s,h) → (b,s/N,h) ← COMM 2: 2×b×s×h bandwidth
    
    x_sp = x_sp + attn_sp  # Residual (both are (b,s/N,h))
    
    # === MLP BLOCK ===
    x_sp = layer_norm(x_sp)  # (b,s/N,h)
    
    # SP→TP transition for MLP
    x_tp = all_gather(x_sp, dim=1)  # (b,s/N,h) → (b,s,h) ← COMM 3: 2×b×s×h bandwidth
    
    # MLP computation in TP region (no communication)
    hidden_local = x_tp @ W_fc1_local  # (b,s,h) → (b,s,4h/N)
    hidden_local = gelu(hidden_local)
    mlp_output = hidden_local @ W_fc2_local  # (b,s,4h/N) → (b,s,h)
    # Note: NO all_reduce here!
    
    # TP→SP transition
    mlp_sp = reduce_scatter(mlp_output, dim=1)  # (b,s,h) → (b,s/N,h) ← COMM 4: 2×b×s×h bandwidth
    
    output_sp = x_sp + mlp_sp  # Residual
    return output_sp  # (b,s/N,h) - stays sharded for next layer

# Total Forward Communication: 2 all_gather + 2 reduce_scatter = 8×b×s×h bandwidth
# Same as TP, but with sequence sharding benefits!
```

## Key Insights

1. **Communication Volume**: Both TP and TP+SP have identical communication volume:
   - **TP**: 4 all_reduces × 2W each = 8×activations bandwidth
   - **TP+SP**: (2 all_gather + 2 reduce_scatter) × 2W each = 8×activations bandwidth

2. **Why "8×activations"**: 
   - 2 all_reduces in forward (attention + MLP) = 4W bandwidth
   - 2 all_reduces in backward (attention + MLP) = 4W bandwidth
   - Total: 8W where W = b×s×h (activation size)

3. **SP Advantages** (despite same communication volume):
   - **Memory Reduction**: Activations are (b,s/N,h) instead of (b,s,h) between TP regions
   - **Better Overlap**: Can overlap SP transitions with computation
   - **LayerNorm Efficiency**: Operates on smaller tensors (b,s/N,h)

4. **Backward Pass Optimization in TP+SP**:
   ```python
   # Standard TP backward (for Q projection):
   dx_partial = dq_local @ W_q_local.T  # Partial gradient
   dx_full = all_reduce(dx_partial)     # Need full gradient
   
   # TP+SP backward (Q,K,V combined):
   dx_partial = dq_local @ W_q_local.T + dk_local @ W_k_local.T + dv_local @ W_v_local.T
   dx_sp = reduce_scatter(dx_partial, dim=1)  # Direct to SP form!
   # Saves one operation by combining all_reduce + scatter into reduce_scatter
   ```

5. **When to Use**:
   - **TP Only**: When sequence length is small, or within-node only
   - **TP+SP**: When activation memory is significant, or when using pipeline parallelism
   - Both are bandwidth-bound and cannot easily overlap with compute

## Communication Bandwidth Summary

| Method | Forward Ops | Backward Ops | Total Bandwidth | Activation Memory |
|--------|------------|--------------|-----------------|-------------------|
| **TP** | 2 all_reduce | 2 all_reduce | 8×b×s×h | b×s×h (full) |
| **TP+SP** | 2 all_gather + 2 reduce_scatter | 2 all_gather + 2 reduce_scatter | 8×b×s×h | b×s×h/N (sharded) |
| **FSDP** | 2L all_gather | L all_gather + L reduce_scatter | 6×params | b×s×h (full) |

Where L = number of layers. Note that FSDP communication is in parameters while TP/SP communication is in activations.