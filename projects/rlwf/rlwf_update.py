# Copyright (c) 2025, Tencent Inc. All rights reserved.
# Data: 2025/1/14 10:45
import os
import tyro
import random
from dataclasses import dataclass, field
from typing import Optional

import torch
import numpy as np

from accelerate import Accelerator
from datasets import Dataset
import transformers
from transformers import LlamaTokenizer
from trl import AutoModelForCausalLMWithValueHead, set_seed, DPOTrainer
from peft import LoraConfig, PeftModel

from utils_dataset import load_data

def setup_seed(seed):
    np.random.seed(seed)
    torch.set_rng_state(torch.manual_seed(seed).get_state())
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def Average(lst):
    return sum(lst) / len(lst)

def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])

def build_dataset(path_to_generated_data, path_to_exp_data):
    dpo_data_dict = load_data(path_to_generated_data, path_to_exp_data)
    ds_train = Dataset.from_dict(dpo_data_dict)

    def return_prompt_and_responses(samples):
        return {
            "prompt": [sample for sample in samples["prompt"]],
            "chosen": samples["chosen"],
            "rejected": samples["rejected"],
        }

    ds_train = ds_train.map(return_prompt_and_responses, batched=True)

    return ds_train


@dataclass
class DPOConfig(transformers.TrainingArguments):
    """
    Arguments related to the DPO training process itself. For all parameters, see: https://huggingface.co/docs/transformers/v4.26.1/en/main_classes/trainer#transformers.TrainingArguments
    """
    beta: Optional[float] = field(
        default=0.1,
        metadata={"help": "The beta factor in DPO loss. Higher beta means less divergence from the initial policy."},
    )
    hub_model_revision: Optional[str] = field(
        default="main",
        metadata={"help": ("The Hub model branch to push the model to.")},
    )
    logging_first_step: bool = field(
        default=True,
        metadata={"help": ("Whether to log and evaluate the first global_step or not.")},
    )
    max_prompt_length: Optional[int] = field(
        default=None,
        metadata={"help": ("For DPO, the maximum length of the prompt to use for conditioning the model.")},
    )
    max_length: Optional[int] = field(
        default=None,
        metadata={"help": ("Used by TRL for reward model training, which tries to read this parameter in init.")},
    )
    optim: Optional[str] = field(default="adamw_hf") # adamw_torch
    remove_unused_columns: bool = field(default=False)
    loss_type: Optional[str] = field(default="sigmoid", metadata={"help": ("The loss type for DPO.")})

def main(args, model_dir):
    output_folder = args.output_dir
    os.makedirs(output_folder, exist_ok=True)

    @dataclass
    class ScriptArguments:
        dpo_config: DPOConfig = DPOConfig(
                output_dir=output_folder,
                beta=args.beta,
                max_length = 512,
                logging_first_step=True, 
                loss_type=args.loss_type, 
                report_to='none',
                seed=args.seed,
                num_train_epochs=args.epoch,
            )
        
        use_seq2seq: bool = False
        """whether to use seq2seq models"""
        
        use_peft: bool = args.use_peft
        """whether to use peft"""
        peft_config: Optional[LoraConfig] = field(
            default_factory=lambda: LoraConfig(
                r=args.lora_r,
                lora_alpha=int(2*args.lora_r),
                bias="none",
                task_type="CAUSAL_LM",
            ),
        )
        trust_remote_code: bool = field(default=False, metadata={"help": "Enable `trust_remote_code`"})
        
        # add to avoid args conflict 
        path_to_neg_data: str = args.path_to_neg_data
        path_to_pos_data: str = args.path_to_pos_data
        model_dir: str = args.model_dir
        loss_type: int = args.loss_type
        beta: float = args.beta
        epoch: int = args.epoch
        lora_r: int = args.lora_r
    
    config = tyro.cli(ScriptArguments) 
        
    # set seed for deterministic eval
    setup_seed(config.dpo_config.seed)
    set_seed(config.dpo_config.seed) 

    # initialize tokenizer
    tokenizer = LlamaTokenizer.from_pretrained(os.path.join(model_dir), legacy=False)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    ds_train = build_dataset(config.path_to_neg_data, config.path_to_pos_data)

    trl_model_class = AutoModelForCausalLMWithValueHead

    # uild the model, the reference model, and the tokenizer.
    if not config.use_peft:
        ref_model = trl_model_class.from_pretrained(model_dir, 
                                                    trust_remote_code=config.trust_remote_code)
        peft_config = None
    else:
        peft_config = config.peft_config
        ref_model = None
    
    # Copy the model to each device
    device_map = {"": Accelerator().local_process_index}
    model = trl_model_class.from_pretrained(
        model_dir,
        trust_remote_code=config.trust_remote_code,
        device_map=device_map,
        peft_config=peft_config
    )

    dpo_trainer = DPOTrainer(model=model, 
                             ref_model=ref_model, 
                             beta=config.dpo_config.beta,
                             loss_type=config.dpo_config.loss_type,
                             args=config.dpo_config,
                             tokenizer=tokenizer, 
                             train_dataset=ds_train,
                             max_length=config.dpo_config.max_length,
                            #  peft_config=peft_config
                            )

    ###############
    # Training loop
    ###############
    print(f"*** Training begin! ***")
    train_result = dpo_trainer.train()
    metrics = train_result.metrics
    metrics["train_samples"] = len(ds_train)
    dpo_trainer.log_metrics("train", metrics)
    dpo_trainer.save_metrics("train", metrics)
    dpo_trainer.save_state()
    print("*** Training complete! ***")

    ##################################
    # Save model
    ##################################
    cur_save_path = os.path.join(output_folder, f'rlwf_checkpoints')
    os.makedirs(cur_save_path, exist_ok=True)
    dpo_trainer.model.save_pretrained(cur_save_path)


if __name__ == "__main__":
    @dataclass
    class ExpConfig:
        model_dir: str = field(default="./checkpoints/")
        output_dir: str= field(default="./resutls/")
        path_to_neg_data: str=field(default="./data/example_generated_sequence.csv")
        path_to_pos_data: str=field(default="./data/example_exp_pos_sequence.csv")
        loss_type: str = field(default='sigmoid')
        beta: float = field(default=0.1)
        use_peft: bool = field(default=True)
        epoch: int = field(default=1)
        lora_r: int = field(default=8)
        seed: int = field(default=0)
    args = tyro.cli(ExpConfig)

    assert args.loss_type in ['sigmoid', 'hinge', 'ipo', 'kto_pair']
    main(model_dir = args.model_dir, args = args)
