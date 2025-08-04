import os
import ray
import socket
import subprocess
import sys
from functools import wraps


def find_free_port():
    """Find a free port for master_port"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]


def ray(num_nodes=1, gpus_per_node=1):
    """
    Simple Ray + torchrun wrapper.
    
    Usage:
        @ray(num_nodes=2, gpus_per_node=1)  
        def main(args):
            # training logic here
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # If already launched by torchrun, just run the function
            if "LOCAL_RANK" in os.environ:
                return func(*args, **kwargs)
            
            # Use Ray to launch torchrun on each node
            master_addr = ray.util.get_node_ip_address()
            master_port = find_free_port()
            
            @ray.remote(num_cpus=1)
            def run_torchrun(node_rank):
                script_path = sys.modules[func.__module__].__file__
                cmd = [
                    "torchrun",
                    f"--nproc_per_node={gpus_per_node}",
                    f"--nnodes={num_nodes}",
                    f"--node_rank={node_rank}",
                    f"--master_addr={master_addr}",
                    f"--master_port={master_port}",
                    script_path
                ] + sys.argv[1:]
                
                return subprocess.run(cmd, check=True)
            
            # Launch torchrun on all nodes
            futures = [run_torchrun.remote(i) for i in range(num_nodes)]
            return ray.get(futures)
            
        return wrapper
    return decorator 