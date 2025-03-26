# Copyright (c) 2025, Tencent Inc. All rights reserved.
# Data: 2025/1/9 15:58
# Author: chenchenqin
from .base_tokenizer import AbstractTokenizer
from ..utils.registry import Registry

TOKENIZER_REGISTRY = Registry("tokenizer")

_GLOBAL_TOKENIZER = None


def get_tokenizer():
    """Return tokenizer."""
    assert _GLOBAL_TOKENIZER is not None, f"init tokenizer first"


def set_tokenizer(cfg):
    global _GLOBAL_TOKENIZER
    _GLOBAL_TOKENIZER = build_tokenizer(cfg)


def create_tokenizer(cls_name: str, path: str, **kwargs) -> AbstractTokenizer:
    return TOKENIZER_REGISTRY[cls_name](path, **kwargs)


def build_tokenizer(cfg):
    name = cfg.tokenizer.name
    tokenizer = TOKENIZER_REGISTRY[name](cfg)
    return tokenizer
