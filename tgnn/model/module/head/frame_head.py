# Copyright (c) 2025, Tencent Inc. All rights reserved.
# Data: 2025/1/9 15:58
# Author: chenchenqin
import torch
import torch.nn as nn

from tgnn.model.layer import Linear
from tgnn.transform import quanternion3_to_4
from tgnn.transform.affine import Rigid, Rotation


class FrameHead(nn.Module):
    """
    Implements part of Algorithm 23.

    Args:
        decouple: whether decouple rotation and translation linear
    """

    def __init__(self, c_s, dof=3, decouple=False):
        """
        Args:
            c_s: Single representation channel dimension
        """
        super(FrameHead, self).__init__()
        assert dof in (3, 4, 6), f"only support quaternion vector size 3 or 4, or ortho6d vector"
        self.c_s = c_s
        self.decouple = decouple
        self.dof = dof
        if decouple:
            self.rotation_linear = Linear(self.c_s, dof, init="final")
            self.translation_linear = Linear(self.c_s, 3, init="final")
        else:
            self.linear = Linear(self.c_s, dof + 3, init="final")

    def forward(self, s: torch.Tensor) -> Rigid:
        """
        Args:
            s； [*, c_s] single representation

        Returns:
            [*, dof + 3] update vector
        """
        if self.decouple:
            rotations = self.rotation_linear(s)
            translations = self.translation_linear(s)
        else:
            rot = self.linear(s)
            rotations = rot[..., :self.dof]
            translations = rot[..., self.dof:]

        if self.dof in (3, 4):
            quanterions = rotations
            if self.dof == 3:
                quanterions = quanternion3_to_4(quanterions)
            return Rigid(Rotation(rot_mats=None, quats=quanterions), translations)
        else:
            return Rigid.from_tensor_9(ortho6d=rotations, trans=translations)
