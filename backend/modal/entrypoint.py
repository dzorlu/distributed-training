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

# Read config.yaml from known locations
_HERE = Path(__file__).resolve().parent
_CANDIDATE_PATHS = [
    _HERE / "config.yaml",
    Path("/workspace/code/backend/modal/config.yaml"),
]
_ENV_PATH = os.environ.get("MODAL_CONFIG_PATH")
if _ENV_PATH:
    _CANDIDATE_PATHS.insert(0, Path(_ENV_PATH))

_cfg: dict = {}
for _p in _CANDIDATE_PATHS:
    if _p.exists():
        with _p.open("r") as f:
            _cfg = yaml.safe_load(f) or {}
        break
if not _cfg:
    raise FileNotFoundError(
        f"Missing Modal config. Checked: {', '.join(str(p) for p in _CANDIDATE_PATHS)}."
    )

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
# Bake the local repo into the image at build time (reproducible submission)
image = image.add_local_dir(str(_REPO_ROOT), "/workspace/code")

    # Optional: declare secret names you have configured in Modal
def _try_secret(name: str):
    try:
        return modal.Secret.from_name(name)
    except Exception:
        return None


SECRETS = list(filter(None, [_try_secret(n) for n in SECRET_NAMES]))

@app.function(
    gpu=GPU_SPEC,
    image=image,
    secrets=SECRETS if SECRETS else None,
    timeout=86400,
)
def run_torchrun(args_json: str):  # type: ignore
    os.environ["RUNNING_UNDER_MODAL"] = "1"
    os.chdir("/workspace/code")

    args = json.loads(args_json)
    script_argv: List[str] = args.get("argv", [])

    # Install GPU-dependent extension at runtime (GPU available now)
    try:
        subprocess.run(
            [
                "python",
                "-m",
                "pip",
                "install",
                "--no-build-isolation",
                "git+https://github.com/fanshiqing/grouped_gemm@main",
            ],
            check=True,
        )
    except Exception:
        pass

    # Prefer calling the file path to avoid Python -m import path issues
    cmd = [
        "torchrun",
        f"--nproc-per-node={GPU_SPEC.split(':', 1)[1] if ':' in GPU_SPEC else '1'}",
        "/workspace/code/train.py",
        *script_argv,
    ]

    completed = subprocess.run(cmd, check=False)
    return completed.returncode


