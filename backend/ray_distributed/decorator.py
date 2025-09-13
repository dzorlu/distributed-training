import os
import socket
import subprocess
import signal
import time
import sys
from functools import wraps

import ray

from parallelization.logging import logger


def find_free_port():
    """Find a free port for torchrun master coordination."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]


def ray_distributed(num_nodes=1, gpus_per_node=1):
    """
    Interactive distributed training launcher using Ray + torchrun.

    Usage:
        @ray_distributed(num_nodes=2, gpus_per_node=4)
        def train():
            ...
        train()
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # If we're already in a worker process, run directly
            if "LOCAL_RANK" in os.environ:
                return func(*args, **kwargs)

            # Single GPU case - no coordination needed
            if num_nodes == 1 and gpus_per_node == 1:
                return func(*args, **kwargs)

            # Initialize Ray cluster if not already connected
            if not ray.is_initialized():
                head_ip = os.environ.get('RAY_HEAD_IP', 'localhost')
                ray.init(address=f"{head_ip}:6379")

            nodes = [node for node in ray.nodes() if node['Alive']]
            if len(nodes) < num_nodes:
                raise RuntimeError(f"Need {num_nodes} nodes, but only {len(nodes)} available")

            master_addr = ray.util.get_node_ip_address()
            master_port = find_free_port()

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
                        self.proc = subprocess.Popen(cmd, preexec_fn=os.setsid)
                        rc = self.proc.wait()
                        norm_rc = rc if rc >= 0 else 128 + abs(rc)
                        if norm_rc in (130, 143):
                            logger.info(f"🛑 Node {self.node_rank} interrupted (rc={rc})")
                        else:
                            logger.info(f"✅ Node {self.node_rank} exited with code {rc}")
                        return rc
                    finally:
                        self.proc = None

                def stop(self):
                    if self.proc and self.proc.poll() is None:
                        try:
                            os.killpg(os.getpgid(self.proc.pid), signal.SIGINT)
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

            runners = [NodeRunner.remote(i) for i in range(num_nodes)]
            script_args = sys.argv[1:] if len(sys.argv) > 1 else []
            futures = [
                runner.run_torchrun.remote(master_addr, master_port, module_name, script_args)
                for runner in runners
            ]

            try:
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



