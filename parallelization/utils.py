import torch
from torch._utils import _get_available_device_type, _get_device_module
import torch.distributed as dist
from torch.distributed.tensor import DTensor

from .logging import logger

def get_device_info() -> tuple[str, torch.device]:
    device_type = _get_available_device_type() or "cuda"
    device_module = _get_device_module(device_type)  # default device_module:torch.cuda
    return device_type, device_module



device_type, device_module = get_device_info()

def describe_group(pg):
    backend = dist.get_backend(pg)
    size = dist.get_world_size(pg)
    global_rank = dist.get_rank()
    group_rank = dist.get_rank(pg)
    members = [dist.distributed_c10d.get_global_rank(pg, r) for r in range(size)]
    return {
        "backend": str(backend),
        "size": size,
        "global_rank": global_rank,
        "group_rank": group_rank,
        "global_ranks_in_group": members,
    }


def log_tensor(label: str, t):
    if isinstance(t, DTensor):
        tl = t.to_local()
        try:
            mesh = t.device_mesh
        except Exception:
            mesh = None
        logger.info(
            f"{label}: DTensor "
            f"global_shape={tuple(t.shape)}, "
            f"local_shape={tuple(tl.shape)}, "
            f"placements={t.placements}, "
            f"mesh={mesh}, "
            f"local_dtype={tl.dtype}, "
            f"local_device={tl.device}"
        )
    else:
        logger.info(
            f"{label}: Tensor "
            f"shape={tuple(t.shape)}, "
            f"dtype={t.dtype}, "
            f"device={t.device}"
        )