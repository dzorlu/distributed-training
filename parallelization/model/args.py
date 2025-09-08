from dataclasses import dataclass
from typing import Optional
from torch.distributed.device_mesh import DeviceMesh

@dataclass
class ModelArgs:
    dim: int = 2048 #2048
    n_layers: int = 16 #64
    n_heads: int = 16
    n_kv_heads: Optional[int] = None
    vocab_size: int = 1000  # defined later by tokenizer
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-3
    gradient_accumulation_steps: int = 4


    batch_size: int = 4
    max_batch_size: int = 32
    max_seq_len: int = 2048
    # If `True`, then each transfor
    # mer block init uses its layer ID, and if
    # `False`, each uses the total number of transformer blocks
    depth_init: bool = True

    # moe
    num_experts: int = 4
    top_k: int = 2
    use_moe: bool = True
    score_func: str = "sigmoid"

    # lr
    lr: float = 0.00005

    # device mesh
    device_mesh: Optional[DeviceMesh] = None