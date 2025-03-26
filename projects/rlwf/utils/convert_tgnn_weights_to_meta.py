import os
import gc
import argparse

import torch
from typing import Dict
from pathlib import Path

def convert_tgnn_checkpoint_to_meta(
        input_tgnn_ckpt = "checkpoints/tgnn",
        output_meta_ckpt = "checkpoints/llama",
        model_size = "7B",
        dtype = torch.float32):
    
    os.makedirs(output_meta_ckpt, exist_ok=True)
    output_meta_ckpt_folder = os.path.join(output_meta_ckpt, model_size)
    os.makedirs(output_meta_ckpt_folder, exist_ok=True)
    output_meta_ckpt = os.path.join(output_meta_ckpt_folder,'consolidated.00.pth')

    checkpoint_input = torch.load(input_tgnn_ckpt, map_location="cpu")
    checkpoint = checkpoint_input['module']

    for name, param in checkpoint.items():
        
        if "c_attn" not in name:
            continue

        # Turn [Q1, Q2, ..., K1, K2, .., V1, V2, ...] into [Q1, K1, V1, Q2, K2, V2, ...] 
        src_chunk_len = param.shape[0]
        
        mat_len = src_chunk_len // 3
        
        dst_chunk_len = mat_len
        
        attn = torch.clone(param)
        
        for j in range(3):
            param[j * mat_len: (j + 1) * mat_len] = attn [j * dst_chunk_len : j * dst_chunk_len + mat_len]
        del attn
   
    gc.collect()
    
    with open(os.path.join(output_meta_ckpt_folder, 'input_model_param.txt'),"w") as current_file:
        for key in checkpoint.keys():
            current_file.write(f"{key}\n")

    converted = convert_state_dict(checkpoint, dtype=dtype)
    
    gc.collect()

    torch.save(converted, Path(output_meta_ckpt))

def convert_state_dict(state_dict: Dict[str, torch.Tensor],
                       dtype: torch.dtype = torch.float32) -> Dict[str, torch.Tensor]:
    converted = {}
    converted["tok_embeddings.weight"] = state_dict["transformer.wte.weight"].to(dtype)
    converted["output.weight"] = state_dict["lm_head.weight"] .to(dtype)
    converted["norm.weight"] = state_dict["transformer.ln_f.weight"].to(dtype)
    
    for key in [k for k in state_dict if k.startswith("transformer.h")]:
        
        layer_idx = key.split(".")[2]

        qkv_size = state_dict["transformer.h.0.attn.c_attn.weight"].size()[0] // 3

        # attention
        converted[f"layers.{layer_idx}.attention.wq.weight"] = state_dict[f"transformer.h.{layer_idx}.attn.c_attn.weight"][:qkv_size].to(dtype)
        converted[f"layers.{layer_idx}.attention.wk.weight"] = state_dict[f"transformer.h.{layer_idx}.attn.c_attn.weight"][qkv_size:-qkv_size].to(dtype)
        converted[f"layers.{layer_idx}.attention.wv.weight"] = state_dict[f"transformer.h.{layer_idx}.attn.c_attn.weight"][-qkv_size:].to(dtype)

        converted[f"layers.{layer_idx}.attention.wo.weight"] = state_dict[f"transformer.h.{layer_idx}.attn.c_proj.weight"].to(dtype)
        # mlp
        converted[f"layers.{layer_idx}.feed_forward.w1.weight"] = state_dict[f"transformer.h.{layer_idx}.mlp.c_fc1.weight"].to(dtype)
        converted[f"layers.{layer_idx}.feed_forward.w2.weight"] = state_dict[f"transformer.h.{layer_idx}.mlp.c_proj.weight"].to(dtype)
        converted[f"layers.{layer_idx}.feed_forward.w3.weight"] = state_dict[f"transformer.h.{layer_idx}.mlp.c_fc2.weight"].to(dtype)
        # rms norm
        converted[f"layers.{layer_idx}.attention_norm.weight"] = state_dict[f"transformer.h.{layer_idx}.rms_1.weight"].to(dtype)
        converted[f"layers.{layer_idx}.ffn_norm.weight"] = state_dict[f"transformer.h.{layer_idx}.rms_2.weight"].to(dtype)
   
    return converted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        help="Location of tgnn weights",
    )
    parser.add_argument(
        "--model_size",
        default=None,
        help="'model size",
    )
    parser.add_argument(
        "--output_dir",
        help="Location to write HF model and tokenizer",
    )
    args = parser.parse_args()
    convert_tgnn_checkpoint_to_meta(input_tgnn_ckpt = args.input_dir,
                                    output_meta_ckpt =args.output_dir,
                                    model_size = args.model_size)

if __name__ == "__main__":
    main()

