import os
import ray
import socket
import subprocess
import sys
from functools import wraps


def find_free_port():
    """Find a free port for torchrun master coordination."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]


def ray_distributed(num_nodes=1, gpus_per_node=1):
    """
    Interactive distributed training launcher using Ray + torchrun.
    
    This allows you to SSH into the head node and launch distributed training
    with a single Python call, rather than manually coordinating torchrun
    across multiple nodes.
    
    Usage:
        @ray_distributed(num_nodes=2, gpus_per_node=4)
        def train():
            # Your training code here - runs on each GPU process
            import torch.distributed as dist
            rank = dist.get_rank()
            local_rank = int(os.environ['LOCAL_RANK'])
            print(f"Process {rank}, local rank {local_rank}")
    
        # Run interactively from head node:
        train()
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Check if we're already in a distributed process
            if "LOCAL_RANK" in os.environ:
                return func(*args, **kwargs)
            
            # Single GPU case - no coordination needed
            if num_nodes == 1 and gpus_per_node == 1:
                return func(*args, **kwargs)
            
            # Initialize Ray cluster if not already connected
            if not ray.is_initialized():
                # Connect to existing Ray cluster
                head_ip = os.environ.get('RAY_HEAD_IP', 'localhost')
                ray.init(address=f"{head_ip}:6379")
            
            # Get all available nodes
            nodes = [node for node in ray.nodes() if node['Alive']]
            if len(nodes) < num_nodes:
                raise RuntimeError(f"Need {num_nodes} nodes, but only {len(nodes)} available")
            
            # Use head node as master for torchrun coordination
            master_addr = ray.util.get_node_ip_address()
            master_port = find_free_port()
            
            # Get module info and arguments
            # When running with python -m, func.__module__ returns '__main__'
            # We need to get the actual module name from __spec__
            if func.__module__ == '__main__':
                main_module = sys.modules['__main__']
                print(f"spec: {main_module.__spec__}")
                module_name = main_module.__spec__.name
            else:
                module_name = func.__module__
            
            print(f"🚀 Launching distributed training:")
            print(f"   Nodes: {num_nodes}")
            print(f"   GPUs per node: {gpus_per_node}")
            print(f"   Total processes: {num_nodes * gpus_per_node}")
            print(f"   Master: {master_addr}:{master_port}")
            print(f"   Module: {module_name}")
            
            @ray.remote(num_gpus=gpus_per_node, max_restarts=1)
            class NodeRunner:
                def __init__(self, node_rank):
                    self.node_rank = node_rank
                
                def run_torchrun(self, master_addr, master_port, module_name, args):
                    """Run torchrun on this node."""
                    cmd = [
                        "torchrun",
                        f"--nproc_per_node={gpus_per_node}",
                        f"--nnodes={num_nodes}",
                        f"--node_rank={self.node_rank}",
                        f"--master_addr={master_addr}",
                        f"--master_port={master_port}",
                        "-m", module_name
                    ] + args
                    
                    print(f"Node {self.node_rank}: {' '.join(cmd)}")
                    
                    try:
                        # Run without capturing output so we can see training progress in real-time
                        result = subprocess.run(cmd, check=True)
                        print(f"✅ Node {self.node_rank} completed successfully")
                        return result
                    except subprocess.CalledProcessError as e:
                        print(f"❌ Node {self.node_rank} failed with exit code {e.returncode}")
                        raise
            
            # Create runners for each node
            # Ray will distribute these across available nodes
            runners = [NodeRunner.remote(i) for i in range(num_nodes)]
            
            script_args = sys.argv[1:] if len(sys.argv) > 1 else []
            
            # Launch torchrun on all nodes
            futures = [
                runner.run_torchrun.remote(master_addr, master_port, module_name, script_args)
                for runner in runners
            ]
            
            try:
                # Wait for all nodes to complete
                results = ray.get(futures)
                print("✅ Distributed training completed successfully!")
                return results
            except Exception as e:
                print(f"❌ Distributed training failed: {e}")
                raise
            
        return wrapper
    return decorator


# Example usage:
if __name__ == "__main__":
    @ray_distributed(num_nodes=2, gpus_per_node=4)
    def train():
        """Example training function."""
        import torch
        import torch.distributed as dist
        
        # Initialize distributed training
        dist.init_process_group(backend='nccl')
        
        rank = dist.get_rank()
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = dist.get_world_size()
        
        # Set device
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')
        
        print(f"Process {rank}/{world_size}, local rank {local_rank}, device {device}")
        
        # Your training code here...
        
        dist.destroy_process_group()
    
    # Launch interactively from head node
    train()