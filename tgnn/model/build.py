# Copyright (c) 2025, Tencent Inc. All rights reserved.
# Data: 2025/1/9 15:58
# Author: chenchenqin
import torch

from tgnn.utils import Registry, get_torch_version, get_logger

MODEL_REGISTRY = Registry("model")  # noqa F401 isort:skip

loger = get_logger()


def build_model(cfg):
    """build model by architecture name
    """
    arch = cfg.model.arch
    assert arch in MODEL_REGISTRY, f"{arch} not in model registry"
    model: torch.nn.Module = MODEL_REGISTRY.get(arch)(cfg)
    model.to(getattr(torch, cfg.solver.dtype))

    if cfg.model.compile.enabled:
        loger.info("set model compiled")
        assert get_torch_version() >= [2, 0], f"only pytorch 2.0 support model compile, get {torch.__version__}"
        model = torch.compile(model, backend=cfg.model.compile.backend)

    return model
