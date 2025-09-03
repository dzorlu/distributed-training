import logging
import os
import torch.distributed as dist

logger = logging.getLogger(__name__)

class RankFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        # Always attach a rank; do NOT filter anything out
        try:
            rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else -1
        except Exception:
            rank = -1
        record.rank = rank
        #return rank == 0 
        return True

def init_logger(level=logging.INFO):
    # Clear existing handlers to avoid duplicate logs when re-initializing
    logger.handlers.clear()
    logger.propagate = False  # prevent double logging via root

    logger.setLevel(level)
    ch = logging.StreamHandler()
    ch.setLevel(level)

    # Use %(rank)d instead of %(dist.get_rank())d
    formatter = logging.Formatter(
        "[rank=%(rank)d pid=%(process)d] %(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    ch.setFormatter(formatter)

    # Attach filter that injects rank into every record
    ch.addFilter(RankFilter())
    logger.addHandler(ch)

    # suppress verbose torch.profiler logging
    os.environ["KINETO_LOG_LEVEL"] = "5"