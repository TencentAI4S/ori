# Copyright (c) 2025, Tencent Inc. All rights reserved.
# Data: 2025/1/9 15:58
# Author: chenchenqin
import logging
import numbers

import torch
import torch.nn as nn

try:
    from apex.normalization.fused_layer_norm import mixed_dtype_fused_rms_norm_affine

    logging.warning("using apex fused rms norm")
except:
    mixed_dtype_fused_rms_norm_affine = None


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization

    Ref：
        1.Root Mean Square Layer Normalization
    """

    def __init__(self,
                 normalized_shape: int,
                 eps: float = 1e-6,
                 device=None,
                 dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.eps = eps
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = torch.Size(normalized_shape)
        self.weight = nn.Parameter(torch.ones(*normalized_shape, **factory_kwargs))

    def _norm(self, hidden_states):
        variance = hidden_states.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
        return hidden_states * torch.rsqrt(variance + self.eps)

    def forward(self, input):
        if mixed_dtype_fused_rms_norm_affine is None or torch.jit.is_tracing() or torch.jit.is_scripting() or not input.is_cuda:
            input = self._norm(input).type_as(input)
            return self.weight * input
        else:
            return mixed_dtype_fused_rms_norm_affine(input, self.weight, self.normalized_shape, self.eps)

    def extra_repr(self):
        return "{normalized_shape}, eps={eps}, " "elementwise_affine={elementwise_affine}".format(**self.__dict__)
