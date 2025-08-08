from .llama2_model import Transformer
from .llama2_model import ModelArgs
import argparse
import os


# Import the ray and profiler decorators
from parallelization import ray, profiler, flop_counter
from parallelization.profiler.decorator import step_profiler
from parallelization.profiler.utils import log_parameter_count


from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel, parallelize_module



def main(args):

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    init_device_mesh("cuda", world_size)

    model_args = ModelArgs()
    model = Transformer.from_model_args(model_args)
    log_parameter_count(model)

    layer_tp_sp_plan = {
        
    }










if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch TP/SP example")
    parser.add_argument("--num-nodes", type=int, default=1, help="Number of nodes for distributed training")
    parser.add_argument("--gpus-per-node", type=int, default=1, help="Number of GPUs per node")
    
    # Profiler arguments
    parser.add_argument("--profile", action="store_true", default=False, help="Enable PyTorch profiler")
    args = parser.parse_args()
    
    # Apply decorators dynamically based on CLI args
    training_func = main
    
    # Apply profiler decorator if requested
    if args.profile:
        training_func = profiler(enabled=True)(training_func)
    
    # Apply ray decorator for distributed training
    distributed_main = ray(num_nodes=args.num_nodes, gpus_per_node=args.gpus_per_node)(training_func)
    distributed_main(args)