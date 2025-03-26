# Copyright (c) 2025, Tencent Inc. All rights reserved.
# Data: 2025/1/14 10:45
import pandas as pd


def load_data(file_generated_data, file_exp_positive_data, threshold=0.6):
    df = pd.read_csv(file_generated_data)

    perplexity_threshold = df['perplexity'].quantile(threshold)
    plddt_threshold = df['plddt'].quantile(threshold)
    neg_df = df[(df['perplexity'] <= perplexity_threshold) & (df['plddt'] >= plddt_threshold)]

    pos_df = pd.read_csv(file_exp_positive_data)

    keys = set(pos_df['prompt']) & set(neg_df['prompt'])

    neg_seqs = {}
    pos_seqs = {}
    for key in keys:
        neg_seqs[key] = neg_df[neg_df['prompt'] == key]['seq'].tolist()
        pos_seqs[key] = pos_df[pos_df['prompt'] == key]['seq'].tolist()

    prompts = []
    chosens = []
    rejecteds = []
    for key in keys:
        if len(neg_seqs[key]) > 0 and len(pos_seqs[key]) > 0:
            for chosen in pos_seqs[key]:
                for reject in neg_seqs[key]:
                    prompts.append(key)
                    chosens.append(chosen)
                    rejecteds.append(reject)

    dpo_data_dict = {
        "prompt": prompts,
        "chosen": chosens,
        "rejected": rejecteds,
    }

    return dpo_data_dict
