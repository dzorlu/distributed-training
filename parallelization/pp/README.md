


- save memory: it shards params as well as activations.
- good communication properties: it is in the order `b x s x h`. It depends on activations.
 - batch size is critical to hide the size of the bubble (e.g. more micro batches)
- dual pipe (zero-bubble pipeline). Separete W and B. 