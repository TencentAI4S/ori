# Copyright (c) 2025, Tencent Inc. All rights reserved.
# Data: 2025/1/9 15:58
# Author: chenchenqin
import os
import sys
import urllib
from pathlib import Path
from typing import Any, List, Callable, Optional, Union
from urllib.request import urlopen

import deepspeed
import torch
from torch.hub import get_dir, download_url_to_file, HASH_REGEX


def get_tokenizer_path(url: str, progress: bool = True, check_hash: bool = True) -> str:
    r"""donwload file at the given URL.

    Args:
        url (str): URL of the object to download
        progress (bool, optional): whether or not to display a progress bar to stderr.
            Default: True
    """
    hub_dir = get_dir()
    model_dir = os.path.join(hub_dir, 'tokenizers')
    os.makedirs(model_dir, exist_ok=True)
    with urlopen(url) as response:
        filename = response.headers.get_filename()

    cached_file = f"{model_dir}/{filename}"
    if not os.path.exists(cached_file):
        sys.stderr.write(f'Downloading: "{url}" to {cached_file}\n')
        hash_prefix = None
        if check_hash:
            r = HASH_REGEX.search(filename)  # r is Optional[Match[str]]
            hash_prefix = r.group(1) if r else None
        download_url_to_file(url, cached_file, hash_prefix, progress=progress)

    return cached_file


def get_model_path(url, progress: bool = True, check_hash: bool = True):
    model_dir = f"{torch.hub.get_dir()}/checkpoints"
    os.makedirs(model_dir, exist_ok=True)
    with urlopen(url) as response:
        filename = response.headers.get_filename()

    cached_file = f"{model_dir}/{filename}"
    if not os.path.exists(cached_file):
        sys.stderr.write(f'Downloading: "{url}" to {cached_file}\n')
        hash_prefix = None
        if check_hash:
            r = HASH_REGEX.search(filename)  # r is Optional[Match[str]]
            hash_prefix = r.group(1) if r else None
        download_url_to_file(url, cached_file, hash_prefix, progress=progress)

    return cached_file


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


def checkpoint_blocks(
        blocks: Union[List[Callable], torch.nn.ModuleList],
        args: Any,
        interval: Optional[int] = 1,
        checkpoint_func=deepspeed.checkpointing.checkpoint,
        return_intervals=False
):
    """
    Chunk a list of blocks and run each chunk with activation
    checkpointing. We define a "block" as a callable whose only inputs are
    the outputs of the previous block.

    Implements Subsection 1.11.8

    Args:
        blocks:
            List of blocks
        args:
            Tuple of arguments for the first block.
        interval:
            Size of each chunk. A higher value corresponds to fewer
            checkpoints, and trades memory for speed. If None, no checkpointing
            is performed.
    Returns:
        The output of the final block
    """

    def wrap(a):
        return (a,) if type(a) is not tuple else a

    def exec(b, a):
        for block in b:
            a = wrap(block(*a))
        return a

    def chunker(s, e):
        def exec_sliced(*a):
            return exec(blocks[s:e], a)

        return exec_sliced

    # Avoids mishaps when the blocks take just one argument
    args = wrap(args)

    if interval is None or not torch.is_grad_enabled():
        return exec(blocks, args)
    elif interval < 1 or interval > len(blocks):
        raise ValueError("blocks_per_ckpt must be between 1 and len(blocks)")

    if return_intervals:
        interval_outputs = []
        for s in range(0, len(blocks), interval):
            e = s + interval
            args = checkpoint_func(chunker(s, e), *args)
            interval_outputs.append(args)
            args = wrap(args)
        return interval_outputs

    for s in range(0, len(blocks), interval):
        e = s + interval
        args = checkpoint_func(chunker(s, e), *args)
        args = wrap(args)

    return args
