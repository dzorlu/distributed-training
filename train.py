
import argparse
import os
import sys
import re
import time
import importlib
from typing import List
import wandb
from datetime import timedelta
from contextlib import nullcontext

# Import backends lazily when chosen to avoid hard dependency costs
from parallelization.profiler.decorator import performance_monitor
from parallelization.profiler.utils import log_parameter_count
from parallelization.model.llama2_model import Transformer
from parallelization.model.llama2_model import ModelArgs
from parallelization.model.moe import GroupedExpert
from parallelization.logging import logger, init_logger
from transformers import AutoTokenizer

from parallelization.utils import device_type, device_module
from torch._utils import _get_available_device_type, _get_device_module



import torch
import torch.nn.functional as F
from torch.distributed.tensor.debug import CommDebugMode
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh, DeviceMesh
from torch.distributed.tensor.parallel import (
    parallelize_module,
)
import torch.distributed._functional_collectives as funcol
import torch.distributed.distributed_c10d as c10d
from torch.distributed.elastic.multiprocessing.errors import record
from torch.distributed.tensor import Shard, Replicate
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor.experimental import context_parallel
from torch.distributed.tensor.experimental._attention import set_rotate_method
from torch.optim import Adam



from parallelization.dataset import get_hf_dataloader

def get_device_info() -> tuple[str, torch.device]:
    device_type = _get_available_device_type() or "cuda"
    device_module = _get_device_module(device_type)  # default device_module:torch.cuda
    logger.info(f"device_type: {device_type}, device_module: {device_module}")
    return device_type, device_module

def apply_parallelization(
    model: torch.nn.Module,
    model_name: str,
    mesh: DeviceMesh,
    model_args: ModelArgs,
    rank: int
):
    """Dynamically import and apply the parallelization plan from the model's directory."""
    try:
        module_path = f"parallelization.{model_name}.parallelize"
        parallelize_module_import = importlib.import_module(module_path)
        parallelize_fn = getattr(parallelize_module_import, "parallelize")
        # This function will now apply the parallelization in-place on the model
        parallelize_fn(model, mesh, model_args, rank)
    except (ModuleNotFoundError, AttributeError) as e:
        raise ImportError(f"Could not import or find 'parallelize' function in '{module_path}'. Make sure the module and function exist.") from e


def extract_unique_param_names(model):
    """Extract unique parameter patterns without layer numbers"""
    patterns = []
    
    for name, param in model.named_parameters():
        # Remove .weight, .bias suffix
        name_without_suffix = '.'.join(name.split('.')[:-1])
        
        # Remove layers.X. pattern if it exists
        pattern = re.sub(r'^layers\.\d+\.', '', name_without_suffix)
        
        patterns.append(pattern)
    
    # Get unique patterns and preserve order
    unique_patterns = []
    seen = set()
    for pattern in patterns:
        if pattern not in seen:
            unique_patterns.append(pattern)
            seen.add(pattern)
    
    return unique_patterns

def rescale_accumulated_loss(unwrapped_loss_fn, accumulation_steps: int):
    """Return a loss function that divides by accumulation_steps.

    This ensures that summing the per-microbatch losses over N steps yields
    the mean loss, and gradients accumulated across those N backward calls
    equal the average gradient.
    """
    def _wrapped(*args, **kwargs):
        return unwrapped_loss_fn(*args, **kwargs) / accumulation_steps
    return _wrapped

@record
def main(args):
    # Flight recorder setup via environment variables
    os.environ["TORCH_NCCL_DUMP_ON_TIMEOUT"] = "1"
    os.environ["TORCH_FR_BUFFER_SIZE"] = "2000"
    os.environ["TORCH_FR_DUMP_TEMP_FILE"] = "/tmp/nccl_trace"
    
    init_logger()

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ.get("RANK", str(local_rank)))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    device_type, device_module = get_device_info()
    torch.cuda.set_device(local_rank)   # set by torchrun
    device = torch.device(f"{device_type}:{int(os.environ['LOCAL_RANK'])}")
    # Device has to be set before creating TorchFT manager.
    device_module.set_device(device)


    logger.info(f"device: {device}")
    logger.info(f"local_rank: {local_rank}, rank: {rank}, world_size: {world_size}")

    # Initialize the default PG with an explicit device mapping
    if dist.is_initialized():
        dist.destroy_process_group()
    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=60),
        device_id=local_rank,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)


    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    dp_size = args.dp_size

    if args.use_cp and args.use_moe:
        raise ValueError("CP and MoE are not supported together")

    ########################
    # Mesh initialization  #
    ########################

    if args.use_moe:
        ep_size = args.ep_size
        dp_shard = dp_size  # treat provided dp_size as the total DP sharding degree
        if dp_shard % ep_size != 0:
            raise ValueError(
                f"EP must divide DP sharding: dp_shard({dp_shard}) % ep({ep_size}) == 0 required."
            )

        dp_shard_in_ep = ep_size                # borrowed by EP (forms EP groups)
        dp_shard_mod_ep = dp_shard // ep_size   # leftover for FSDP sharding. borrowing!

        # With only DP+EP (tp=cp=pp=1), the total WORLD_SIZE equals dp_shard
        if dp_shard != world_size:
            raise ValueError(
                f"With DP2EP and tp=cp=pp=1, WORLD_SIZE({world_size}) must equal dp_shard({dp_shard})"
            )

        # Create the 2D mesh
        device_mesh = init_device_mesh(
            "cuda",
            (dp_shard_mod_ep, dp_shard_in_ep),
            mesh_dim_names=("dp_shard", "ep"),
        )

        # Create aliases for DP
        # DP will be used for data loading
        device_mesh[("dp_shard", "ep")]._flatten(mesh_dim_name="dp")

        dp_mesh  = device_mesh["dp"]     # size = R*C = 8  -> used for non-MoE FSDP
        row_mesh = device_mesh["dp_shard"]   # size = R   = 4  -> used for expert FSDP (inside each column)
        col_mesh = device_mesh["ep"]   # size = C   = 2  -> used for EP ownership + a2a
    
    elif args.use_cp:

        # Create the 2D mesh
        # FSDP is still on dp_size because cp regions need to be all-reduced as well -
        # CP itself doesn’t add a model-wide grad all-reduce.
        # If dp=1 and cp>1 without FSDP, you’d train unsynchronized replicas (not desirable).
        device_mesh = init_device_mesh(
            "cuda",
            (args.dp_size, args.cp_size),
            mesh_dim_names=("dp_shard", "cp"),
        )

        device_mesh[("dp_shard", "cp")]._flatten(mesh_dim_name="dp")

        dp_mesh = device_mesh["dp_shard"]   # size = R   = 4  -> used for data parallelism. 
        row_mesh  = device_mesh["dp"]     # size = R*C = 8  -> used for FSDP
        col_mesh = device_mesh["cp"]   # size = C   = 2  -> used for CP

    else:
        # FSDP
        device_mesh = init_device_mesh(
            "cuda",
            (args.dp_size, args.cp_size),
            mesh_dim_names=("dp_shard")
        )
        row_mesh  = device_mesh["dp_shard"]
        dp_mesh  = device_mesh["dp_shard"] 


    logger.info(f"{device_mesh=}")
    assert torch.cuda.current_device() == local_rank

    rank = device_mesh.get_rank()
    if args.wandb:
        if rank == 0:
            wandb.init(project=args.dataset_name)
            wandb.define_metric("prof/step", hidden=True)
            wandb.define_metric("prof/*", step_metric="prof/step")


    # Use len(tokenizer) to get the actual vocabulary size including all special tokens
    actual_vocab_size = len(tokenizer)
    logger.info(f"Tokenizer vocab_size attribute: {tokenizer.vocab_size}")
    logger.info(f"Actual tokenizer length: {actual_vocab_size}")

    model_args = ModelArgs(use_moe=args.use_moe, vocab_size=actual_vocab_size)
    
    # This context allows you to define a model's architecture and a
    # ll its parameters without allocating any memory for the weights, 
    model_args.device_mesh = device_mesh
    with torch.device("meta"):
        model = Transformer.from_model_args(model_args)
    log_parameter_count(model, model_args)

    if args.use_moe:
        # ----- 3) Attach EP to routed experts (columns mesh) -----
        #   • partitions expert tensors across columns (ownership)
        #   • installs dispatch/combine (a2a) hooks on forward
        apply_parallelization(
            model=model,
            model_name=args.model_name,
            mesh=col_mesh,
            model_args=model_args,
            rank=local_rank
        )


        # ----- 4) FSDP for MoE on row (4-way) -----
        for layer_id, transformer_block in enumerate(model.layers):
            if transformer_block.moe_enabled and row_mesh:
                fully_shard(
                    transformer_block.feed_forward.experts,
                    mesh=row_mesh,
                )

        # ----- 5) FSDP for non-MoE on dp (8-way) -----
        # Instead of wrapping the whole model and then overriding, be explicit:
        #   - find and wrap ONLY the non-MoE modules on the 8-way dp mesh
        for layer_id, transformer_block in enumerate(model.layers):
            fully_shard(
                transformer_block,
                mesh=dp_mesh,
            )

        # ----- 5) Wrap everything else on dp (8-way) -----
        fully_shard(model, mesh=dp_mesh)

    elif args.use_cp:
        # https://docs.pytorch.org/tutorials/unstable/context_parallel.html
        # https://discuss.pytorch.org/t/distributed-w-torchtitan-breaking-barriers-training-long-context-llms-with-1m-sequence-length-in-pytorch-using-context-parallel/215082
        set_rotate_method('allgather')

        # ----- 5) FSDP for CP on dp (8-way) -----
        for layer_id, transformer_block in enumerate(model.layers):
            fully_shard(
                transformer_block,
                mesh=row_mesh,
            )

        # ----- 5) Wrap everything else on dp (8-way) -----
        fully_shard(model, mesh=row_mesh)
    else:
        # ----- 1) FSDP
        for layer_id, transformer_block in enumerate(model.layers):
            fully_shard(
                transformer_block,
                mesh=row_mesh,
            )

        # ----- 2) Wrap everything else on dp (8-way) -----
        fully_shard(model, mesh=row_mesh)

 


    logger.info(f"Model after parallelization {model=}\n")
    logger.info("=== Parameter Analysis After Parallelization ===")
    regular_tensors = []
    dtensors = []

    for name, param in model.named_parameters():
        if hasattr(param, 'placements'):  # DTensor
            placement = getattr(param, "placements", None)
            logger.info(f"{name=}, {placement=}")
            dtensors.append(name)
        else:  # Regular tensor
            regular_tensors.append(name)
            
    logger.info(f"DTensors: {len(dtensors)}")
    logger.info(f"Regular tensors: {len(regular_tensors)}")
    if regular_tensors:
        logger.info(f"{device=} Regular tensor parameters:")
        for name in regular_tensors:
            logger.info(f"{device=}, {name=}")


    model.to_empty(device=device)
    # The weights are initialized directly on each target GPU
    # after the model has been parallelized
    with torch.no_grad():
        logger.info(f"init_weights")
        model.init_weights()

    # foreach=False is not optimized.
    optimizer = Adam(model.parameters(), lr=model_args.lr)
    global_step = 0

    monitor = performance_monitor(
        model,
        enabled=args.profile,
        flop_counter_step=2,
        comm_logger_step=2,
    )


    # Build a scaled loss function for gradient accumulation
    def _base_loss_fn(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            y.view(-1),
            ignore_index=tokenizer.pad_token_id,
        )

    loss_fn = rescale_accumulated_loss(
        _base_loss_fn, model_args.gradient_accumulation_steps
    )

    @monitor
    def training_step(x, y, i):
        ctx = nullcontext()
        if args.use_cp:
            ctx = context_parallel(
                col_mesh,
                buffers=[x, y, model.freqs_cis],
                buffer_seq_dims=[1, 1, 0],
                no_restore_buffers={x, y},
            )
        with ctx:
            logits = model(x)
            if torch.isnan(logits).any():
                logger.warning(f"NaN in logits")
            _loss = loss_fn(logits, y)

            # Collect auxiliary losses from all MoE layers
            if args.use_moe:
                aux_losses = []
                for layer in model.layers:
                    if hasattr(layer.feed_forward.router, 'aux_loss') and layer.feed_forward.router.aux_loss is not None:
                        aux_losses.append(layer.feed_forward.router.aux_loss)
                aux_losses = torch.stack(aux_losses).sum()
                logger.info(f"{i=}, {aux_losses=}, {_loss=}")
                _loss += aux_losses

            
            del logits  # avoid peaking memory at the start of the backward pass.
            _loss.backward()
            return _loss

    # --- Data Loading ---
    logger.info(f"{args.dataset_name=}, {args.dataset_config_name=}, {args.dataset_split=}")
    dataloader = get_hf_dataloader(
        dataset_name=args.dataset_name,
        dataset_config_name=args.dataset_config_name,
        dataset_split=args.dataset_split,
        tokenizer=tokenizer,
        model_args=model_args,
        device_mesh=dp_mesh,
    )

    # Zero once before the first accumulation window
    optimizer.zero_grad(set_to_none=True)
    accumulated_losses: List[torch.Tensor] = []
    accumulated_n_tokens_seen = 0
    lapsed_in_sec = 0
    for i, batch in enumerate(dataloader, 1):
        t = time.time()
        x = batch['input_ids'].to(device, non_blocking=True)
        #logger.info(f"{x.shape=}")
        y = batch['labels'].to(device, non_blocking=True)
        loss = training_step(x, y, i)
        lapsed_in_sec +=  time.time() - t 

        accumulated_losses.append(loss.detach())
        accumulated_n_tokens_seen += y.numel()
        
        if i % model_args.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            # Sum of n_tokens_seen equals mean n_tokens_seen over the accumulation window
            log_n_tokens_seen = torch.tensor(accumulated_n_tokens_seen, device=device, dtype=torch.long)
            reduceOp = c10d.ReduceOp.SUM.name
            log_n_tokens_seen = funcol.all_reduce(log_n_tokens_seen, reduceOp=reduceOp, group=dp_mesh)

            # secs per step
            lapsed_in_sec = torch.tensor(lapsed_in_sec, device=device, dtype=torch.long)
            reduceOp = c10d.ReduceOp.AVG.name
            lapsed_in_sec = funcol.all_reduce(lapsed_in_sec, reduceOp=reduceOp, group=dp_mesh)

        
            # Sum of scaled losses equals mean loss over the accumulation window
            log_loss = torch.sum(torch.stack(accumulated_losses))
            reduceOp = c10d.ReduceOp.AVG.name
            log_loss = funcol.all_reduce(log_loss, reduceOp=reduceOp, group=dp_mesh)

            if args.wandb and rank == 0:
                wandb.log(
                    {
                        "loss": log_loss.item(),
                        "n_tokens_seen": log_n_tokens_seen.item(),
                        "sec_per_step": lapsed_in_sec,
                    },
                    step=global_step, 
                    commit=True
                )
                global_step += 1
            accumulated_losses = []
            lapsed_in_sec = 0

    # Flush leftover microbatches if the last window is incomplete
    if accumulated_losses:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        log_loss = torch.sum(torch.stack(accumulated_losses))
        reduceOp = c10d.ReduceOp.AVG.name
        log_loss = funcol.all_reduce(log_loss, reduceOp=reduceOp, group=dp_mesh)
        if args.wandb and rank == 0:
            wandb.log(
                {

                    "loss": log_loss.item(),
                    "n_tokens_seen": log_n_tokens_seen.item(),
                },
                step=global_step,
                commit=True,
            )
            global_step += 1

    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch Distributed Training")
    parser.add_argument("--use-moe", action="store_true", help="Whether to use MoE")
    parser.add_argument("--use-cp", action="store_true", help="Whether to use Context Parallel")
    parser.add_argument("--model-name", type=str, default="model", help="The name of the model folder under /parallelization")
    parser.add_argument("--dp-size", type=int, default=1, help="Data parallel size")
    parser.add_argument("--ep-size", type=int, default=1, help="Expert parallel size for MoE models")
    parser.add_argument("--cp-size", type=int, default=1, help="Conetext parallel size")
    parser.add_argument("--num-nodes", type=int, default=1, help="Number of nodes for distributed training")
    parser.add_argument("--tokenizer-name", type=str, default="Qwen/Qwen-tokenizer", help="The name of the tokenizer to use")
    parser.add_argument("--dataset-name", type=str, default="wikitext", help="Hugging Face dataset name")
    #parser.add_argument("--dataset-name", type=str, default="zai-org/LongAlign-10k", help="Hugging Face dataset name")
    #parser.add_argument("--dataset-config-name", type=str, default="wikitext-2-v1", help="Hugging Face dataset config name (e.g., 'en', 'wikitext-2-raw-v1')")
    #parser.add_argument("--dataset-config-name", type=str, help="Hugging Face dataset config name (e.g., 'en', 'wikitext-2-raw-v1')")
    parser.add_argument("--dataset-split", type=str, default="train", help="Dataset split to use (e.g., 'train', 'train[:1%]')")
    
    # Profiler arguments
    parser.add_argument("--profile", action="store_true", default=False, help="Enable PyTorch profiler")
    parser.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=True, help="Enable W&B logging by default (use --no-wandb to disable)")
    parser.add_argument("--backend", type=str, choices=["ray", "modal"], default="ray", help="Execution backend")

    
    args = parser.parse_args()
    
    # Apply decorators dynamically based on CLI args
    training_func = main
    
    # Select backend (CLI overrides env)
    backend = args.backend or os.environ.get("BACKEND", "ray")

    # If running under Modal or explicitly selecting modal, run directly (no wrapper)
    if backend == "modal" or os.environ.get("RUNNING_UNDER_MODAL") == "1":
        if args.num_nodes and args.num_nodes != 1:
            raise ValueError("Modal backend currently supports single-node only; set --num-nodes 1")
        distributed_main = training_func
    else:
        from backend.ray_distributed.decorator import ray_distributed
        distributed_main = ray_distributed(num_nodes=args.num_nodes, gpus_per_node=args.gpus_per_node)(training_func)
    try:
        distributed_main(args)
    except KeyboardInterrupt as e:
        logger.error(f"Error: {e}")
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
            logger.info("Process group destroyed")
            wandb.finish()