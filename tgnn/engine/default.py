# Copyright (c) 2025, Tencent Inc. All rights reserved.
# Data: 2025/1/13 10:28
# Author: chenchenqin
import torch

from tgnn.utils import seed_all_rng


def setup(cfg):
    torch.set_grad_enabled(False)
    torch.set_float32_matmul_precision("high")
    seed_all_rng(cfg.rng_seed)
