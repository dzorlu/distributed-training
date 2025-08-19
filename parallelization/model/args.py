from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelArgs:
    dim: int = 4096 #2048
    n_layers: int = 64 #64
    n_heads: int = 16
    n_kv_heads: Optional[int] = None
    vocab_size: int = 1000  # defined later by tokenizer
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5

    max_batch_size: int = 32
    max_seq_len: int = 32768
    # If `True`, then each transfor
    # mer block init uses its layer ID, and if
    # `False`, each uses the total number of transformer blocks
    depth_init: bool = True

    # moe
    num_experts: int = 8
    top_k: int = 2