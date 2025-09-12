import os
import json
import subprocess
from typing import List

import modal
import yaml
from pathlib import Path


# Define the Modal app and image only when running under Modal CLI/runtime
APP_NAME = "distributed-training"
app = modal.App(APP_NAME)

# Read config.yaml colocated with this file
_HERE = Path(__file__).resolve().parent
_CONFIG_PATH = _HERE / "config.yaml"
if not _CONFIG_PATH.exists():
    raise FileNotFoundError(f"Missing Modal config at {_CONFIG_PATH}. Create with 'gpu: H100:8'")
with _CONFIG_PATH.open("r") as f:
    _cfg = yaml.safe_load(f) or {}
GPU_SPEC = _cfg.get("gpu")
if not GPU_SPEC or not isinstance(GPU_SPEC, str):
    raise ValueError("Modal config must contain 'gpu: <TYPE:COUNT>'")
SECRET_NAMES = _cfg.get("secrets", [])
if SECRET_NAMES is None:
    SECRET_NAMES = []
if not isinstance(SECRET_NAMES, list):
    raise ValueError("Modal config 'secrets' must be a list of secret names or omitted")

    # Base image and environment mirroring non-Ray portions of skypilot_small.yaml
image = (
    modal.Image.from_registry("pytorch/pytorch:2.8.0-cuda12.6-cudnn9-devel")
    .apt_install(["git", "tmux", "htop", "iperf3", "netcat-openbsd"])
    .pip_install(
        "tabulate",
        "transformers",
        "datasets",
        "huggingface-hub",
        "wandb",
        "git+https://github.com/fanshiqing/grouped_gemm@main",
    )
    .env(
        {
            "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
            "NCCL_DEBUG": "INFO",
        }
    )
)

    # Local code mount (adjust path if your repo root differs)
# Use a path relative to this file to avoid hardcoded absolute paths
_REPO_ROOT = (_HERE / ".." / "..").resolve()
CODE_MOUNT = modal.Mount.from_local_dir(str(_REPO_ROOT), remote_path="/workspace/code")

    # Optional: declare secret names you have configured in Modal
def _try_secret(name: str):
    try:
        return modal.Secret.from_name(name)
    except Exception:
        return None

SECRETS = list(filter(None, [_try_secret(n) for n in SECRET_NAMES]))

@app.function(
    name="run_torchrun",
    gpu=GPU_SPEC,
    image=image,
    mounts=[CODE_MOUNT],
    secrets=SECRETS if SECRETS else None,
    timeout=0,
)
def run_torchrun(args_json: str):  # type: ignore
    os.environ["RUNNING_UNDER_MODAL"] = "1"
    os.chdir("/workspace/code")

    args = json.loads(args_json)
    script_argv: List[str] = args.get("argv", [])

    # Prefer calling the file path to avoid Python -m import path issues
    cmd = [
        "torchrun",
        f"--nproc-per-node={GPU_SPEC.split(':', 1)[1] if ':' in GPU_SPEC else '1'}",
        "/workspace/code/train.py",
        *script_argv,
    ]

    completed = subprocess.run(cmd, check=False)
    return completed.returncode


