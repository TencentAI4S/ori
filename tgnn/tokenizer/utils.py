# Copyright (c) 2025, Tencent Inc. All rights reserved.
# Data: 2025/1/9 15:58
# Author: chenchenqin
import numpy as np

def fit_vocab_size_to_dtype(vocab_size):
    if vocab_size is not None and vocab_size < 65500:
        return np.uint16
    else:
        return np.int32