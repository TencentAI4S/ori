# Copyright (c) 2025, Tencent Inc. All rights reserved.
# Data: 2025/1/13 14:42
# Author: chenchenqin
from tgnn.model.utils import load_tokenizer_workaround


def main():
    ret = load_tokenizer_workaround("https://drive.google.com/uc?export=download&id=1WGj4dqBPR61FWiqci33_XGMMtLPNllGQ")
    print(ret)


if __name__ == '__main__':
    main()