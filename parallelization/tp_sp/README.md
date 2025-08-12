# PyTorch Tensor Parallelism: Mental Models and Implementation

## Core Mental Model: Ownership and Responsibility

**Think about what each rank owns and when communication happens:**

- **ColwiseParallel**: "I own output columns → I need ALL input columns"
  - Communication: All-gather BEFORE computation
  - Ownership: Complete output neurons

- **RowwiseParallel**: "I own input rows → I produce partial outputs"  
  - Communication: All-reduce AFTER computation
  - Ownership: Subset of input features

## Quick Reference Table

```
+------------------+----------------+----------------+----------------+------------------+
| Style            | Needs Input    | Produces       | Weight Shard   | Communication    |
+------------------+----------------+----------------+----------------+------------------+
| ColwiseParallel  | Replicate()    | Shard(-1)      | Shard(0)*      | All-gather input |
| RowwiseParallel  | Shard(-1)      | Replicate()    | Shard(1)*      | All-reduce output|
| SequenceParallel | Shard(1)       | Shard(1)       | None           | None             |
+------------------+----------------+----------------+----------------+------------------+
* For Linear. Embeddings are special - see below.
```

### ⚠️ Weight Storage Confusion

**Linear layer constructor vs weight shape:**
```python
nn.Linear(in_features=768, out_features=2048)
# But weight is stored as: [out_features, in_features] = [2048, 768]!
#                           ↑ dim=0        ↑ dim=1

# So for ColwiseParallel:
# Shard(0) shards dim 0 of [2048, 768] → the 2048 output features
# Even though 768 comes first in Linear(768, 2048)!
```

## Linear Layer Parallelism

### ColwiseParallel: Own Output Columns

```python
# Linear(in=768, out=2048) but weight stored as [2048, 768]!
#                                                ↑ dim=0
# Shard(0) splits the 2048 output features

# Weight sharding:
# Full weight: [2048, 768] (out_features, in_features)
# Rank 0: weight[0:1024, :]    = [1024, 768]
# Rank 1: weight[1024:2048, :] = [1024, 768]

# Forward pass with shapes:
Input: [B=4, S=512, H=768]          # Need ALL 768 input features
         ↓ (all-gather if needed)
     [4, 512, 768] @ Weight.T
     [4, 512, 768] @ [768, 1024]    # Note transpose!
         ↓
Output: [4, 512, 1024]               # My 1024 output columns
        Shard(-1)                    # Feature-sharded
```

### RowwiseParallel: Own Input Rows

```python
# Linear(in=2048, out=512) but weight stored as [512, 2048]!
#                                                      ↑ dim=1
# Shard(1) splits the 2048 input features

# Weight sharding:
# Full weight: [512, 2048] (out_features, in_features)
# Rank 0: weight[:, 0:1024]    = [512, 1024]
# Rank 1: weight[:, 1024:2048] = [512, 1024]

# Forward pass with shapes:
Input: [4, 512, 1024]                # My 1024 input features
       Shard(-1)                     # From ColwiseParallel
         ↓
     [4, 512, 1024] @ Weight.T
     [4, 512, 1024] @ [1024, 512]   # Partial computation
         ↓
Output: [4, 512, 512] Partial()     # Needs reduction
         ↓ (all-reduce or reduce-scatter)
       [4, 512, 512] Replicate()    # Full output
```

## Embedding Layer Special Cases

Embeddings use **lookup** not matmul, so sharding strategy differs:

RowwiseParallel: Each rank owns COMPLETE ROWS (full embeddings)

Rank 0: Owns tokens 0-15999 with their FULL 768-dim embeddings

Rank 1: Owns tokens 16000-31999 with their FULL 768-dim embeddings

### Embedding RowwiseParallel

```python
# nn.Embedding(vocab_size=32000, embed_dim=768)
# Weight: [32000, 768] → Shard(0) on vocab dimension!
# Rank 0: [0:16000, 768], Rank 1: [16000:32000, 768]

# Forward pass:
Token IDs: [4, 512] integers        # MUST be Replicate()!
           ↓
    Each rank does partial lookup:
    Rank 0: embeddings for tokens 0-16000 (zeros for others)
    Rank 1: embeddings for tokens 16000-32000 (zeros for others)
           ↓ (all-reduce to sum)
Embeddings: [4, 512, 768]           # Complete embeddings
```

**Key insight**: Token IDs are tiny (just integers) so we replicate them. The embedding table is huge, so we shard it.

### Embedding ColwiseParallel

```python
# Weight: [32000, 768] → Shard(1) on embedding dimension
# Rank 0: [32000, 0:384], Rank 1: [32000, 384:768]

# Each rank stores full vocab but partial embedding dims
# Output naturally becomes Shard(-1) feature-sharded
```

## Typical Transformer Configuration

### With Sequence Parallelism

```python
   tp_plan = {
        # === Embeddings ===
        # Transferring lower payload embedding dimension (vs ~40k dimensional payload)
        "tok_embeddings": RowwiseParallel(
            # Token-ids are replicated in each GPU!
            input_layouts=Replicate(),
            # The token embeddings output Shard(1) to maintain consistent input format for all transformer layers.
            # Now EVERY transformer block receives the same format. 
            # the first operation in transformer block is `attention_norm` op which is SP.
            output_layouts=Shard(1),
        ),
        
        # === For each TransformerBlock (order follows forward pass) ===
        


        # 1. First operation: self.attention_norm(x)
        "layers.*.attention_norm": SequenceParallel(),  # Expects Shard(1), outputs Shard(1)

        # 2. Attention module needs input redistribution
        # PrepareModuleInput sees:
        # - Arg 1: DTensor [32, 128, 2048] with Shard(1)
        # - Arg 2: Convert to DTensor [256, 32] with Replicate()

        # After redistribution:
        # - Arg 1: DTensor [32, 256, 2048] with Replicate() 
        # - Arg 2: DTensor [256, 32] with Replicate()

        # Both are now DTensors with consistent global shapes!
        "layers.*.attention": PrepareModuleInput(
            input_layouts=(Shard(1), Replicate()),  # norm output is Shard(1), freqs_cis is 
            desired_input_layouts=(Replicate(), Replicate()),  # wq/wk/wv need Replicate() - ALL-GATHER here
        ),



        # Attention layers  
        # Unlike a regular tensor, a DTensor is aware of the parallelism plans and 
        # will automatically handle changes in the num_heads dimension.
        # The use_local_output=False ensures you get tensors with **global shapes**, 
        # making view operations work correctly without manual num_heads adjustment.
        "layers.*.attention.wq": ColwiseParallel(use_local_output=False),
        "layers.*.attention.wk": ColwiseParallel(use_local_output=False),
        "layers.*.attention.wv": ColwiseParallel(use_local_output=False),
        # Reduce-scatter op here. TP -> SP
        "layers.*.attention.wo": RowwiseParallel(output_layouts=Shard(1)),
        
        # 3. Second operation: self.ffn_norm(h)
        "layers.*.ffn_norm": SequenceParallel(),  # Expects Shard(1), outputs Shard(1)
        
        # 4. FeedForward module needs input redistribution
        "layers.*.feed_forward": PrepareModuleInput(
            input_layouts=(Shard(1),),  # From ffn_norm
            desired_input_layouts=(Replicate(),),  # w1/w3 need Replicate() - ALL-GATHER here
        ),
        
        # Feed forward layers
        # return self.w2(F.silu(self.w1(x)) * self.w3(x))
        "layers.*.feed_forward.w1": ColwiseParallel(),
        "layers.*.feed_forward.w3": ColwiseParallel(),
        # Reduce-scatter op here for norm operations for next layer.
        "layers.*.feed_forward.w2": RowwiseParallel(output_layouts=Shard(1)), 
        
        # === Final model operations ===
        # Norms - all SequenceParallel
        "norm": SequenceParallel(),  # Final norm before output
        
        # Final output layer
        "output": ColwiseParallel(
            input_layouts=Shard(1),  # From final norm (needs to be specified!)
            # so that we don't have to fetch from other ranks. 
            # it is replicated in each GPU
            # for loss calculation
            output_layouts=Replicate()
        ),
    }
```

## Data Flow Through Transformer

```python
Data Flow Through Transformer (2 Ranks)

+----------------------+---------------+-------------+-------------------------+----------------------+---------------+-------------+
| Operation            | Input Tensor  | Input       | Weight Shape & Sharding | Communication        | Output Tensor | Output      |
|                      |               | Layout      |                         | (Input→Output)       |               | Layout      |
+----------------------+---------------+-------------+-------------------------+----------------------+---------------+-------------+
| Token IDs            | [4, 512]      | Replicate() | -                       | -                    | [4, 512]      | Replicate() |
+----------------------+---------------+-------------+-------------------------+----------------------+---------------+-------------+
| tok_embeddings       | [4, 512]      | Replicate() | [32000, 768]            | 1. All-reduce        | [4, 256, 768] | Shard(1)    |
|                      |               |             | Rank 0: [16000, 768]    |    (sum partials)    |               |             |
|                      |               |             | Rank 1: [16000, 768]    | 2. Scatter           |               |             |
|                      |               |             | Shard(0) on vocab       |    (to Shard(1))     |               |             |
+----------------------+---------------+-------------+-------------------------+----------------------+---------------+-------------+
| attention_norm       | [4, 256, 768] | Shard(1)    | [768] weights           | None                 | [4, 256, 768] | Shard(1)    |
|                      |               |             | Replicate()             |                      |               |             |
+----------------------+---------------+-------------+-------------------------+----------------------+---------------+-------------+
| PrepareModuleInput   | [4, 256, 768] | Shard(1)    | -                       | All-gather           | [4, 512, 768] | Replicate() |
| (attention)          |               |             |                         |                      |               |             |
+----------------------+---------------+-------------+-------------------------+----------------------+---------------+-------------+
| attention.wq         | [4, 512, 768] | Replicate() | [768, 768]              | None                 | [4, 512, 384] | Shard(-1)   |
| (Linear)             |               |             | Per rank: [384, 768]    | (local matmul)       |               |             |
|                      |               |             | Shard(0) on out         |                      |               |             |
+----------------------+---------------+-------------+-------------------------+----------------------+---------------+-------------+
| attention.wk         | [4, 512, 768] | Replicate() | [768, 768]              | None                 | [4, 512, 384] | Shard(-1)   |
| (Linear)             |               |             | Per rank: [384, 768]    | (local matmul)       |               |             |
|                      |               |             | Shard(0) on out         |                      |               |             |
+----------------------+---------------+-------------+-------------------------+----------------------+---------------+-------------+
| attention.wv         | [4, 512, 768] | Replicate() | [768, 768]              | None                 | [4, 512, 384] | Shard(-1)   |
| (Linear)             |               |             | Per rank: [384, 768]    | (local matmul)       |               |             |
|                      |               |             | Shard(0) on out         |                      |               |             |
+----------------------+---------------+-------------+-------------------------+----------------------+---------------+-------------+
| Attention compute    | Q,K,V:        | Shard(-1)   | -                       | None                 | [4, 512, 768] | Shard(-1)   |
| (SDPA + heads)       | [4, 512, 384] |             |                         | (local compute)      |               |             |
+----------------------+---------------+-------------+-------------------------+----------------------+---------------+-------------+
| attention.wo         | [4, 512, 768] | Shard(-1)   | [768, 768]              | Reduce-scatter       | [4, 256, 768] | Shard(1)    |
| (Linear)             |               |             | Per rank: [768, 384]    |                      |               |             |
|                      |               |             | Shard(1) on in          |                      |               |             |
+----------------------+---------------+-------------+-------------------------+----------------------+---------------+-------------+
| Residual add         | [4, 256, 768] | Shard(1)    | -                       | None                 | [4, 256, 768] | Shard(1)    |
+----------------------+---------------+-------------+-------------------------+----------------------+---------------+-------------+
| ffn_norm             | [4, 256, 768] | Shard(1)    | [768] weights           | None                 | [4, 256, 768] | Shard(1)    |
|                      |               |             | Replicate()             |                      |               |             |
+----------------------+---------------+-------------+-------------------------+----------------------+---------------+-------------+
| PrepareModuleInput   | [4, 256, 768] | Sh
```


### When Communication Happens

```python
# ColwiseParallel: BEFORE computation
if input != Replicate():
    input = all_gather(input)       # Get full input
output = input @ my_weight_columns  # Compute my outputs

# RowwiseParallel: AFTER computation  
partial = my_input @ my_weight_rows # Compute with my input
if output_layout == Replicate():
    output = all_reduce(partial)    # Sum all partials
elif output_layout == Shard(1):
    output = reduce_scatter(partial) # Sum and shard
```

### Collective Operations

| From → To | Operation | Example |
|-----------|-----------|---------|
| `Shard(1)` → `Replicate()` | All-gather | Attention input |
| `Partial()` → `Replicate()` | All-reduce | RowwiseParallel default |
| `Partial()` → `Shard(1)` | Reduce-scatter | RowwiseParallel with SP |
| `Shard(-1)` → `Shard(1)` | All-to-all | Feature to sequence |

## PrepareModuleInput/Output

Handles mismatches at module boundaries:

```python
# Problem: Attention module receives Shard(1) but wq/wk/wv need Replicate()
# Solution: PrepareModuleInput redistributes ONCE for all internal layers

"attention": PrepareModuleInput(
    input_layouts=(Shard(1), Replicate()),      # What arrives
    desired_input_layouts=(Replicate(), Replicate()), # What's needed
)

# This avoids 3x communication if each of wq/wk/wv did their own all-gather
```

## Key Design Principles

1. **Minimize Communication**: Chain ColwiseParallel → RowwiseParallel for single all-gather + reduce-scatter
2. **Memory vs Compute**: Embeddings optimize memory (RowwiseParallel), output optimizes compute (ColwiseParallel)
3. **Sequence Parallelism**: Keeps activations sharded between layers, reducing memory by factor of world_size
4. **Token IDs**: Always replicated (tiny), enabling distributed vocabulary storage
5. **Loss Computation**: Output typically needs Replicate() or special loss_parallel handling
