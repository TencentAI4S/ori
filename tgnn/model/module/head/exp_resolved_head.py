# Copyright (c) 2025, Tencent Inc. All rights reserved.
# Data: 2025/1/9 15:58
# Author: chenchenqin
import torch.nn as nn

from tgnn.model.layer import Linear


class ExperimentallyResolvedHead(nn.Module):
    """
    For use in computation of "experimentally resolved" loss, subsection
    1.9.10
    Args:
        c_s: Input channel dimension
        c_out: Number of distogram bins
    """

    def __init__(self, c_s, c_out=37):
        super(ExperimentallyResolvedHead, self).__init__()
        self.c_s = c_s
        self.c_out = c_out
        self.linear = Linear(self.c_s, self.c_out, init="final")

    def forward(self, s):
        """
        Args:
            s: [*, N_res, C_s] single embedding
        Returns: [*, N, C_out] logits
        """
        logits = self.linear(s)

        return logits
