import os
import sys
import json
from functools import wraps


def modal_distributed():
    """Launch training on Modal using GPU spec loaded from backend/modal/config.yaml."""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if os.environ.get("RUNNING_UNDER_MODAL") == "1":
                return func(*args, **kwargs)

            from .entrypoint import run_torchrun

            payload = {"argv": sys.argv[1:] if len(sys.argv) > 1 else []}
            args_json = json.dumps(payload)

            # Call the configured Modal function (GPU/count come from config.yaml)
            rc = run_torchrun.remote(args_json).get()
            if rc not in (0, None):
                raise SystemExit(rc)
            return rc

        return wrapper

    return decorator


