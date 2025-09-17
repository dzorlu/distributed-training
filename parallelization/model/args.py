from dataclasses import dataclass
from typing import Optional
from torch.distributed.device_mesh import DeviceMesh

@dataclass
class ModelArgs:
    dim: int = 2048 #2048
    n_layers: int = 8 #64
    n_heads: int = 16
    n_kv_heads: Optional[int] = None
    vocab_size: int = 1000  # defined by tokenizer
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-3
    gradient_accumulation_steps: int = 4


    batch_size: int = 2
    max_batch_size: int = 32
    max_seq_len: int = 4096
    # If `True`, then each transfor
    # mer block init uses its layer ID, and if
    # `False`, each uses the total number of transformer blocks
    depth_init: bool = True

    # moe
    num_experts: int = 8
    top_k: int = 2
    use_moe: bool = True
    score_func: str = "softmax"
    aux_loss_coeff: float = 0.1
    moe_warmup_steps: int = 250
    moe_noise_scale: float = 0.1

    # lr
    lr: float = 0.001

    # device mesh
    device_mesh: Optional[DeviceMesh] = None