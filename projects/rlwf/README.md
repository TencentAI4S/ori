Here's a polished version of your README file using Markdown:

# About

This is the official implementation of the framework "Reinforcement Learning from Wet-lab Feedback" (RLWF) from our
paper "De Novo Design of Functional Proteins with ORI".

## Requirements

- **System**: Linux
- **GPU Memory**: 40GB
- **Python**: 3.8
- **Python Packages**:
  ```sh
  pip install tyro==0.5.17
  pip install numpy==1.24.4
  pip install pandas==2.0.3
  pip install torch==2.0.1
  pip install transformers==4.31.0
  pip install peft==0.6.2
  pip install trl==0.8.6
  ```

### Important

A small change is required in the `trl` package: **In line 990 of `[path to package]/trl/trainer/dpo_trainer.py`,
change `.logits` to `[0]`**. Typically, the `[path to package]` is `/usr/local/python/lib/python3.8/site-packages/`.

## Prepare Data

Please refer to `./data/example_generated_sequence.csv` and `./data/example_exp_pos_sequence.csv` for the format of
generated data and wet-lab data.

## Prepare Model

Please prepare the pre-trained checkpoints of the model in HuggingFace format and place them in
`[path to PGM checkpoints]`.

Here are the steps to convert the TGNN checkpoint into a Hugging Face formatted model. Currently, we only support the
conversion of the 3B model.

1. Convert TGNN format to LLaMA format by running:
   ```sh
   python3 utils/convert_tgnn_weights_to_meta.py --input_dir [path to tgnn checkpoint] --output_dir [path to save llama checkpoint] --model_size '3B'
   ```

2. Copy the `params.json` file to `[path to save llama checkpoint]/[model size]`. We provide the [
   `params.json`](projects/RLWF/utils/params.json) for the 3B model. Run:
   ```sh
   cp utils/params.json [path to save llama checkpoint]/3B/
   ```

3. Convert LLaMA format to Hugging Face format by running:
   ```sh
   python3 utils/convert_llama_weights_to_hf.py --input_dir [path to llama checkpoint] --output_dir [path to save hf checkpoint] --model_size "3B" --tokenizer_path [path to tokenizer model]
   ```

## Running the RLWF

```sh
python rlwf_update.py --model_dir [path to PGM checkpoints]
```

Feel free to replace placeholders like `[path to package]`, `[path to tgnn checkpoint]`,
`[path to save llama checkpoint]`, `[model size]`, and `[path to tokenizer model]` with the actual paths and values
specific to your setup.


