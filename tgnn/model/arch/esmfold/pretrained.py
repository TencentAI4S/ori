# Copyright (c) 2025, Tencent Inc. All rights reserved.
# Data: 2025/1/9 15:58
# Author: chenchenqin
from tgnn.model.utils import load_hub_workaround
from .model import ESMFold


def load_esmfold_model(model_name="esmfold_3B_v1"):
    assert model_name in (
        "esmfold_3B_v0",
        "esmfold_3B_v1",
        "esmfold_structure_module_only_8M",
        "esmfold_structure_module_only_8M_270K",
        "esmfold_structure_module_only_35M",
        "esmfold_structure_module_only_35M_270K",
        "esmfold_structure_module_only_150M",
        "esmfold_structure_module_only_150M_270K",
        "esmfold_structure_module_only_650M",
        "esmfold_structure_module_only_650M_270K",
        "esmfold_structure_module_only_3B",
        "esmfold_structure_module_only_3B_270K",
        "esmfold_structure_module_only_15B"
    )
    url = f"https://dl.fbaipublicfiles.com/fair-esm/models/{model_name}.pt"
    model_data = load_hub_workaround(url)
    cfg = model_data["cfg"]["model"]
    print(model_data["cfg"])
    model_state = model_data["model"]
    model = ESMFold(lddt_hidden_dim=cfg.lddt_head_hid_dim,
                    use_esm_attn_map=cfg.use_esm_attn_map,
                    trunk_config=cfg.trunk)
    expected_keys = set(model.state_dict().keys())
    found_keys = set(model_state.keys())
    missing_essential_keys = []
    for missing_key in expected_keys - found_keys:
        if not missing_key.startswith("esm."):
            missing_essential_keys.append(missing_key)

    if missing_essential_keys:
        raise RuntimeError(f"Keys '{', '.join(missing_essential_keys)}' are missing.")

    model.load_state_dict(model_state, strict=False)
    return model
