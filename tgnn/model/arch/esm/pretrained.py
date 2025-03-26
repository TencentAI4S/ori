# Copyright (c) 2025, Tencent Inc. All rights reserved.
# Data: 2025/1/9 15:58
# Author: chenchenqin
import re
import urllib
import warnings
from pathlib import Path

import torch

from tgnn.tokenizer.alphabet import Alphabet
from .esm2 import ESM2


def load_hub_workaround(url):
    try:
        data = torch.hub.load_state_dict_from_url(url, progress=False, map_location="cpu")
    except RuntimeError:
        # Pytorch version issue - see https://github.com/pytorch/pytorch/issues/43106
        fn = Path(url).name
        data = torch.load(
            f"{torch.hub.get_dir()}/checkpoints/{fn}",
            map_location="cpu",
        )
    except urllib.error.HTTPError as e:
        raise Exception(f"Could not load {url}, check if you specified a correct model name?")
    return data


def load_regression_hub(model_name):
    url = f"https://dl.fbaipublicfiles.com/fair-esm/regression/{model_name}-contact-regression.pt"
    regression_data = load_hub_workaround(url)
    return regression_data


def _has_regression_weights(model_name):
    """Return whether we expect / require regression weights;
    Right now that is all models except ESM-1v, ESM-IF, and partially trained ESM2 models"""
    return not ("esm1v" in model_name or "esm_if" in model_name or "270K" in model_name or "500K" in model_name)


def _download_model_and_regression_data(model_name):
    url = f"https://dl.fbaipublicfiles.com/fair-esm/models/{model_name}.pt"
    model_data = load_hub_workaround(url)
    if _has_regression_weights(model_name):
        regression_data = load_regression_hub(model_name)
    else:
        regression_data = None
    return model_data, regression_data


def check_model_state(model, model_state, regression_data):
    expected_keys = set(model.state_dict().keys())
    found_keys = set(model_state.keys())
    if regression_data is None:
        expected_missing = {"contact_head.regression.weight", "contact_head.regression.bias"}
        error_msgs = []
        missing = (expected_keys - found_keys) - expected_missing
        if missing:
            error_msgs.append(f"Missing key(s) in state_dict: {missing}.")
        unexpected = found_keys - expected_keys
        if unexpected:
            error_msgs.append(f"Unexpected key(s) in state_dict: {unexpected}.")

        if error_msgs:
            raise RuntimeError(
                "Error(s) in loading state_dict for {}:\n\t{}".format(
                    model.__class__.__name__, "\n\t".join(error_msgs)
                )
            )
        if expected_missing - found_keys:
            warnings.warn(
                "Regression weights not found, predicting contacts will not produce correct results."
            )


def load_esm_model(model_name, include_head=True):
    assert model_name in (
        "esm2_t48_15B_UR50D",
        "esm2_t36_3B_UR50D",
        "esm2_t33_650M_UR50D",
        "esm2_t30_150M_UR50D",
        "esm2_t12_35M_UR50D",
        "esm2_t6_8M_UR50D"
    )
    model_data, regression_data = _download_model_and_regression_data(model_name)
    if regression_data is not None:
        model_data["model"].update(regression_data["model"])

    def upgrade_state_dict(state_dict):
        """Removes prefixes 'model.encoder.sentence_encoder.' and 'model.encoder.'."""
        prefixes = ["encoder.sentence_encoder.", "encoder."]
        pattern = re.compile("^" + "|".join(prefixes))
        state_dict = {pattern.sub("", name): param for name, param in state_dict.items()}
        return state_dict

    cfg = model_data["cfg"]["model"]
    print(f"esm config: {cfg}")
    state_dict = model_data["model"]
    state_dict = upgrade_state_dict(state_dict)
    tokenizer = Alphabet.from_architecture(model_name)
    model = ESM2(
        vocab_size=len(tokenizer),
        num_layers=cfg.encoder_layers,
        embedding_dim=cfg.encoder_embed_dim,
        num_heads=cfg.encoder_attention_heads,
        token_dropout=cfg.token_dropout,
        pad_id=tokenizer.pad_id,
        mask_id=tokenizer.mask_id,
        eos_id=tokenizer.eos_id,
        prepend_bos=tokenizer.prepend_bos,
        append_eos=tokenizer.append_eos,
        include_head=include_head
    )
    model.load_state_dict(state_dict, strict=False)
    return model, tokenizer