# Copyright (c) 2025, Tencent Inc. All rights reserved.
# Data: 2025/1/9 15:58
# Author: chenchenqin
from ml_collections import FieldReference

from .config_node import CfgNode as CN

_C = CN()
_C.disk_dir = FieldReference("/mnt/chenchenqin")
_C.experiment_dir = "experiments"
_C.log_dir = _C.get_ref("experiment_dir") + "/logs"
_C.model_dir = _C.get_ref("experiment_dir") + "/models"
_C.cache_dir = _C.get_ref("disk_dir") + "/.cache"
_C.torchhub_dir = _C.get_ref("cache_dir") + "/torch/hub"

_C.rng_seed = 42
_C.log_freq = 1
_C.seq_len = 512
_C.server = CN()
_C.port = 8888

_C.dataset = CN()
_C.dataset.name = ""
_C.dataset.data_dir = "bio_datasets"
_C.dataset.splits = None
_C.dataset.files = None
_C.dataset.seq_len = 256
# -------------------------------tokenizer----------------------#
_C.tokenizer = CN()
_C.tokenizer.name = "sentencepiece"
_C.tokenizer.path = ""
_C.tokenizer.kmer = 3
_C.tokenizer.mapping = None
_C.tokenizer.special_tokens = None
_C.tokenizer.prepend_bos = False
_C.tokenizer.append_eos = False
_C.tokenizer.num_reserved_tokens = 200
# -------------------------------solver-------------------------#
_C.solver = CN()
_C.solver.device = "cuda"
_C.solver.dtype = "float32"  # param dtype
# -------------------------------model-------------------------#
_C.model = CN()
_C.model.arch = "gpt"
_C.model.type = "gpt2"
_C.model.pooler = None
_C.model.num_layers = 6
_C.model.num_kv_heads = None
_C.model.num_heads = 6
_C.model.num_row_kv_heads = None
_C.model.num_col_kv_heads = None
_C.model.num_hiddens = 768
_C.model.dropout = 0.
_C.model.bias = False
_C.model.num_classes = 1
_C.model.from_pretrained = False
_C.model.weights = ""
_C.model.eps = 1e-5
_C.model.packed_swiglu = False
_C.model.sync_batch_norm = False
_C.model.compile = CN({"enabled": False})
_C.model.compile.backend = "inductor"  # hidet
_C.model.compile.mode = None
# v2: flash attention v2
_C.model.attention_mode = "native"
## ----------model.generation---------##
_C.model.generation = CN()
_C.model.generation.method = "topk"
# set top_k=0 and top_p=0 to disable sampling
_C.model.generation.top_k = 192
_C.model.generation.top_p = 0.
_C.model.generation.temperature = 1.0