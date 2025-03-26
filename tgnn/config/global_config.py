# Copyright (c) 2025, Tencent Inc. All rights reserved.
# Data: 2025/1/9 15:58
# Author: chenchenqin
from .default import _C

def get_config(clone=False):
    if clone:
        return _C.clone()

    return _C

