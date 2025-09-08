import os
import ray
import socket
import subprocess
import signal
import time
import sys
from functools import wraps
from ..logging import logger


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
                # In a worker process, just run the user's function
                # The logger will be initialized by the main training script
                return func(*args, **kwargs)

            # --- This section runs on the head node before launching workers ---
            
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
                logger.info(f"spec: {main_module.__spec__}")
                module_name = main_module.__spec__.name
            else:
                module_name = func.__module__
            
            logger.info(f"🚀 Launching distributed training:")
            logger.info(f"   Nodes: {num_nodes}")
            logger.info(f"   GPUs per node: {gpus_per_node}")
            logger.info(f"   Total processes: {num_nodes * gpus_per_node}")
            logger.info(f"   Master: {master_addr}:{master_port}")
            logger.info(f"   Module: {module_name}")
            
            @ray.remote(num_gpus=gpus_per_node, max_restarts=1, max_concurrency=2)
            class NodeRunner:
                def __init__(self, node_rank):
                    self.node_rank = node_rank
                    self.proc = None
                
                def run_torchrun(self, master_addr, master_port, module_name, args):
                    """Run torchrun on this node."""
                    cmd = [
                        "torchrun",
                        f"--nproc_per_node={gpus_per_node}",
                        f"--nnodes={num_nodes}",
                        f"--node_rank={self.node_rank}",
                        f"--master_addr={master_addr}",
                        f"--master_port={master_port}",
                        "-m", module_name,
                        *args,
                    ]
                    
                    logger.info(f"Node {self.node_rank}: {' '.join(cmd)}")
                    
                    try:
                        # Start torchrun in its own process group so we can terminate the whole tree
                        self.proc = subprocess.Popen(cmd, preexec_fn=os.setsid)
                        rc = self.proc.wait()
                        # Normalize negative rc (killed by signal) to 128+signal
                        norm_rc = rc if rc >= 0 else 128 + abs(rc)
                        if norm_rc in (130, 143):  # SIGINT, SIGTERM
                            logger.info(f"🛑 Node {self.node_rank} interrupted (rc={rc})")
                        else:
                            logger.info(f"✅ Node {self.node_rank} exited with code {rc}")
                        return rc
                    finally:
                        self.proc = None

                def stop(self):
                    if self.proc and self.proc.poll() is None:
                        try:
                            # Try graceful KeyboardInterrupt first so worker finally blocks run
                            os.killpg(os.getpgid(self.proc.pid), signal.SIGINT)
                            # Give it a moment to exit gracefully
                            for _ in range(10):
                                if self.proc.poll() is not None:
                                    break
                                time.sleep(0.2)
                            if self.proc.poll() is None:
                                os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
                                for _ in range(10):
                                    if self.proc.poll() is not None:
                                        break
                                    time.sleep(0.2)
                            if self.proc.poll() is None:
                                os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
                        except Exception:
                            pass
            
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
                logger.info("✅ Distributed training completed successfully!")
                return results
            except KeyboardInterrupt:
                logger.info("❌ Distributed training interrupted by user")
                for r in runners:
                    try:
                        ray.get(r.stop.remote(), timeout=2)
                    except Exception:
                        pass
                for f in futures:
                    try:
                        ray.cancel(f, force=True)
                    except Exception:
                        pass
                raise
            except Exception as e:
                logger.error(f"❌ Distributed training failed: {e}")
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
        
        logger.info(f"Process {rank}/{world_size}, local rank {local_rank}, device {device}")
        
        # Your training code here...
        
        dist.destroy_process_group()
    
    # Launch interactively from head node
    train()