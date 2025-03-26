# Copyright (c) 2025, Tencent Inc. All rights reserved.
# Data: 2025/1/9 15:58
# Author: chenchenqin
import logging
import os
import random
from datetime import datetime

import numpy as np
import torch


def seed_all_rng(seed=None):
    """
    Set the random seed for the RNG in torch, numpy and python.

    Args:
        seed (int): if None, will use a strong random seed.
    """
    if seed is None:
        seed = (os.getpid() +
                int(datetime.now().strftime("%S%f")) +
                int.from_bytes(os.urandom(2), "big")
                )
        logger = logging.getLogger(__name__)
        logger.info("Using a generated random seed {}".format(seed))

    if seed is not None and seed > 0:
        np.random.seed(seed)
        torch.set_rng_state(torch.manual_seed(seed).get_state())
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
    else:
        raise ValueError('Seed ({}) should be a positive integer.'.format(seed))


def get_torch_version():
    return tuple(int(x) for x in torch.__version__.split(".")[:2])