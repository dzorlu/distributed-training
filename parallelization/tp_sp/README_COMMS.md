

https://github.com/pytorch/examples/tree/main/distributed/tensor_parallelism
https://docs.pytorch.org/tutorials/intermediate/TP_tutorial.html






Communication pattern (excluding MLP layer for simplicity)
```
+---------------+------------------------+---------------------------+-----------------------------+----------------+
| Component     | Weight Sharding        | Forward Pass              | Backward Pass               | Memory per GPU |
+---------------+------------------------+---------------------------+-----------------------------+----------------+
| Q projection  | ColwiseParallel        | No communication          | all_reduce(dx)              | 1/N weights    |
|               | W[:h/N,:] per GPU      | (b,s,h) → (b,s,h/N)       | dx_partial:(b,s,h)          |                |
|               |                        |                           | → dx_full:(b,s,h)           |                |
+---------------+------------------------+---------------------------+-----------------------------+----------------+
| K projection  | ColwiseParallel        | No communication          | all_reduce(dx)              | 1/N weights    |
|               | W[:h/N,:] per GPU      | (b,s,h) → (b,s,h/N)       | dx_partial:(b,s,h)          |                |
|               |                        |                           | → dx_full:(b,s,h)           |                |
+---------------+------------------------+---------------------------+-----------------------------+----------------+
| V projection  | ColwiseParallel        | No communication          | all_reduce(dx)              | 1/N weights    |
|               | W[:h/N,:] per GPU      | (b,s,h) → (b,s,h/N)       | dx_partial:(b,s,h)          |                |
|               |                        |                           | → dx_full:(b,s,h)           |                |
+---------------+------------------------+---------------------------+-----------------------------+----------------+
| Attention ops | No weights             | No communication          | No communication            | -              |
|               |                        | (b,s,h/N) → (b,s,h/N)     | (b,s,h/N) → (b,s,h/N)       |                |
+---------------+------------------------+---------------------------+-----------------------------+----------------+
| O projection  | RowwiseParallel        | all_reduce(output)        | No communication            | 1/N weights    |
|               | W[:,:h/N] per GPU      | y_partial:(b,s,h)         | dX:(b,s,h/N) → (b,s,h/N)    |                |
|               |                        | → y_full:(b,s,h)          |                             |                |
+---------------+------------------------+---------------------------+-----------------------------+----------------+
| TOTAL         | All weights 1/N        | 1 all_reduce              | 3 all_reduces               | 25% (N=4)      |
+---------------+------------------------+---------------------------+-----------------------------+----------------+
```
Weight grad in backward computation is local. So need to gather weights to flow the gradient back.

Tensor parallelismalso does help reduce activation memory for the matrix multiplications since the intermediate activations are sharded across GPUs.

Yet, ops like `LayerNorm`, `RMSNorm`, `Dropout` require full `h` for each step in thesequence. Further, `AR` is a bottleneck - the program needs to wait for all GPUs to proceed after `o` projection. `sequence parallelism` comes to help.

With `sequence parallelism`, you replace the `all-reduce` operation at the of the TP region with `reduce-scatter` from `b,s,h` to `b,s/tp,h`. This reduces communication to only `b·s·h/tp` per GPU vs all-reduce's `b·s·h·(tp-1)/tp`. Further, the `SP` region now operation.


```
+---------------+----------+------------------------+---------------------------+-----------------------------+----------------+
| Component     | Region   | Weight Sharding        | Forward Pass              | Backward Pass               | Memory per GPU |
+---------------+----------+------------------------+---------------------------+-----------------------------+----------------+
| LayerNorm     | SP       | No sharding            | No communication          | No communication            | Full weights   |
|               |          | (small params)         | Input: (b,s/N,h)          | Operates on (b,s/N,h)       |                |
+---------------+----------+------------------------+---------------------------+-----------------------------+----------------+
| SP→TP trans   | -        | -                      | all_gather(activations)   | reduce_scatter(grads)       | -              |
|               |          |                        | (b,s/N,h) → (b,s,h)       | (b,s,h) → (b,s/N,h)         |                |
+---------------+----------+------------------------+---------------------------+-----------------------------+----------------+
| Q projection  | TP       | ColwiseParallel        | No communication          | No communication            | 1/N weights    |
|               |          | W[:D/N, :] per GPU     | Input: (b,s,h) → (b,s,h/N)| Weight grad: local          |                |
+---------------+----------+------------------------+---------------------------+-----------------------------+----------------+
| K projection  | TP       | ColwiseParallel        | No communication          | No communication            | 1/N weights    |
|               |          | W[:D/N, :] per GPU     | Input: (b,s,h) → (b,s,h/N)| Weight grad: local          |                |
+---------------+----------+------------------------+---------------------------+-----------------------------+----------------+
| V projection  | TP       | ColwiseParallel        | No communication          | No communication            | 1/N weights    |
|               |          | W[:D/N, :] per GPU     | Input: (b,s,h) → (b,s,h/N)| Weight grad: local          |                |
+---------------+----------+------------------------+---------------------------+-----------------------------+----------------+
| Attention ops | TP       | No weights             | No communication          | No communication            | -              |
|               |          |                        | Works on (b,s,h/N)        | Gradients stay sharded      |                |
+---------------+----------+------------------------+---------------------------+-----------------------------+----------------+
| O projection  | TP       | RowwiseParallel        | No communication          | No communication            | 1/N weights    |
|               |          | W[:, :D/N] per GPU     | (b,s,h/N) → (b,s,h)       | Weight grad: local          |                |
|               |          |                        | (output NOT reduced)      | (input grad NOT scattered)  |                |
+---------------+----------+------------------------+---------------------------+-----------------------------+----------------+
| TP→SP trans   | -        | -                      | reduce_scatter(output)    | all_gather(grads)           | -              |
|               |          |                        | (b,s,h) → (b,s/N,h)       | (b,s/N,h) → (b,s,h)         |                |
+---------------+----------+------------------------+---------------------------+-----------------------------+----------------+
| TOTAL         |          | Attention: 1/N         | 1 all_gather +            | 1 reduce_scatter +          | ~25% (N=4)     |
|               |          |                        | 1 reduce_scatter          | 1 all_gather                |                |
+---------------+----------+------------------------+---------------------------+-----------------------------+----------------+
```







https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=sequence_parallelism

Taking MLP layer into account as well, in the forward pass with vanilla TP we had two **all-reduce** operations per transformer block, and in SP we have two **all-gather** and two **reduce-scatter** operations per transformer block. So, SP does twice the number of communication operations as TP. But since an all-reduce operation can be broken down into an all-gather and a reduce-scatter , they’re actually equivalent in terms of communication cost.

For Q,K,V layers, `all-reduce` step in backward computation is replaces with `reduces-scatter` during the `SP->TP` transition. Specifically;

```
# Backward Pass  
def backward_sp_to_tp_with_q_projection(dq_local, x_full, W_q_local):
    """
    dq_local: (b, s, h/tp) - gradient from attention on each GPU
    x_full: (b, s, h) - saved from forward pass
    W_q_local: (h, h/tp) - column-sharded Q weights
    """
    
    # Q projection backward
    # Each GPU computes PARTIAL gradient w.r.t input
    dx_partial = dq_local @ W_q_local.T  # (b, s, h/tp) @ (h/tp, h) → (b, s, h)
    # NOTE: This is partial! In standard TP, we'd do all_reduce(dx_partial) here
    
    # SP→TP transition backward (this is the gradient of all_gather)
    # reduce_scatter BOTH sums the partials AND scatters
    dx_sp = reduce_scatter(dx_partial, dim=1)  # (b, s, h) → (b, s/tp, h)
    # This single op replaces: dx_full = all_reduce(dx_partial)
    #                         dx_sp = scatter(dx_full)
    
    # Weight gradient (local computation, no communication needed)
    dW_q_local = x_full.T @ dq_local  # (h, b*s) @ (b*s, h/tp) → (h, h/tp)
    
    return dx_sp, dW_q_local


# Without SP (standard TP only):
def backward_q_projection_standard_tp(dq_local, x_replicated, W_q_local):
    """For comparison: standard TP needs all_reduce"""
    dx_partial = dq_local @ W_q_local.T  # (b, s, h/tp) @ (h/tp, h) → (b, s, h)
    dx_replicated = all_reduce(dx_partial)  # Sum partials, keep replicated!
    return dx_replicated  # (b, s, h) on ALL GPUs - wasteful!
```


Just like vanilla `TP`, `TP+SP` can’t easily be overlapped with compute, which makes throughput heavily dependent on the communication bandwidth.