# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import torch.distributed as dist


logger = logging.getLogger()


class RankFilter(logging.Filter):
    def filter(self, record):
        rank = dist.get_rank() if dist.is_initialized() else 0
        record.rank = rank
        if record.levelno == logging.INFO:
            return rank == 0
        return True


def init_logger():
    # Clear existing handlers to avoid duplicate logs
    if logger.hasHandlers():
        logger.handlers.clear()

    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "[rank%(rank)d] %(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    ch.setFormatter(formatter)
    logger.addFilter(RankFilter())
    logger.addHandler(ch)

    # suppress verbose torch.profiler logging
    os.environ["KINETO_LOG_LEVEL"] = "5"
