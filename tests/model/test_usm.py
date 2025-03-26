# Copyright (c) 2025, Tencent Inc. All rights reserved.
# Data: 2025/2/20 19:11
# Author: chenchenqin
import torch

from tgnn.config import get_config
from tgnn.model import build_model


def main():
    cfg = get_config()
    model_path = "/mnt/jianhuayao/ori_release/usm_100m.pt"
    state = torch.load(model_path, map_location=torch.device("cpu"))
    model_state = state["module"]
    cfg.update(state["config"])
    model = build_model(cfg)
    model.load_state_dict(model_state)
    model.eval()
    batch_x = torch.randint(0, 16, [8, 512])
    print(batch_x)
    output = model(batch_x)
    for k, v in output.items():
        if isinstance(v, torch.Tensor):
            print(k, v)


if __name__ == '__main__':
    main()
