# Copyright (c) 2025, Tencent Inc. All rights reserved.
# Data: 2025/1/9 15:58
# Author: chenchenqin
import io
import json
import os
import shutil
import sys
from collections.abc import Mapping
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
import wget


def flatten_dict(results):
    """
    Expand a hierarchical dict of scalars into a flat dict of scalars.
    If results[k1][k2][k3] = v, the returned dict will have the entry
    {"k1/k2/k3": v}.

    Args:
        results (dict):
    """
    r = {}
    for k, v in results.items():
        if isinstance(v, Mapping):
            v = flatten_dict(v)
            for kk, vv in v.items():
                r[k + "/" + kk] = vv
        else:
            r[k] = v
    return r


def to(tensor,
       device=None,
       dtype=None,
       non_blocking=False):
    if isinstance(tensor, (list, tuple)):
        new_tensors = []
        for t in tensor:
            new_tensors.append(to(t,
                                  device=device,
                                  dtype=dtype,
                                  non_blocking=non_blocking))
        return new_tensors

    elif isinstance(tensor, dict):
        new_dict = {}
        for name, value in tensor.items():
            new_dict[name] = to(value,
                                device=device,
                                dtype=dtype,
                                non_blocking=non_blocking)
        return new_dict
    elif isinstance(tensor, torch.Tensor):
        return tensor.to(device=device,
                         dtype=dtype,
                         non_blocking=non_blocking)
    elif isinstance(tensor, np.ndarray):
        return torch.tensor(tensor, dtype=dtype, device=device)
    else:
        return tensor


def to_device(tensor, device, non_blocking=False):
    return to(tensor, device, non_blocking=non_blocking)


def to_cuda(tensor, non_blocking=False):
    if isinstance(tensor, (list, tuple)):
        new_tensors = []
        for t in tensor:
            new_tensors.append(to_cuda(t, non_blocking=non_blocking))
        return new_tensors

    elif isinstance(tensor, dict):
        new_dict = {}
        for name, value in tensor.items():
            new_dict[name] = to_cuda(value, non_blocking=non_blocking)

        return new_dict
    elif isinstance(tensor, torch.Tensor):
        return tensor.cuda(non_blocking=non_blocking)
    else:
        return tensor


def to_cpu(tensor):
    if isinstance(tensor, (list, tuple)):
        new_tensors = []
        for t in tensor:
            new_tensors.append(to_cpu(t))

        return new_tensors
    elif isinstance(tensor, dict):
        new_dict = {}
        for name, value in tensor.items():
            new_dict[name] = to_cpu(value)

        return new_dict
    elif isinstance(tensor, torch.Tensor):
        return tensor.cpu()
    else:
        return tensor


def to_numpy(tensor):
    if isinstance(tensor, (list, tuple)):
        new_tensors = []
        for t in tensor:
            new_tensors.append(to_numpy(t))

        return new_tensors
    elif isinstance(tensor, dict):
        new_dict = {}
        for name, value in tensor.items():
            new_dict[name] = to_numpy(value)

        return new_dict
    elif isinstance(tensor, torch.Tensor):
        return tensor.cpu().numpy()
    else:
        return tensor


def to_list(tensor):
    if isinstance(tensor, (torch.Tensor, np.ndarray)):
        return tensor.tolist()

    return tensor


def record_stream(tensor):
    if isinstance(tensor, (list, tuple)):
        new_tensors = []
        for t in tensor:
            new_tensors.append(record_stream(t))
        return new_tensors
    elif isinstance(tensor, dict):
        new_dict = {}
        for name, value in tensor.items():
            new_dict[name] = record_stream(value)
        return new_dict
    elif isinstance(tensor, torch.Tensor):
        return tensor.record_stream(torch.cuda.current_stream())
    else:
        return tensor


def to_size(batch_x, target_size, mode='nearest'):
    if isinstance(batch_x, (list, tuple)):
        new_tensors = []
        for t in batch_x:
            new_tensors.append(to_size(t, target_size, mode))

        return new_tensors
    elif isinstance(batch_x, dict):
        new_dict = {}
        for name, value in batch_x.items():
            new_dict[name] = to_size(value, target_size, mode)

        return new_dict
    elif isinstance(batch_x, torch.Tensor):
        batch_x = F.interpolate(batch_x, target_size, mode=mode)

        return batch_x
    else:
        # TODO: add numpy array resize
        return batch_x


def clone(tensor):
    if isinstance(tensor, (list, tuple)):
        new_tensors = []
        for t in tensor:
            new_tensors.append(clone(t))

        return new_tensors
    elif isinstance(tensor, dict):
        new_dict = {}
        for name, value in tensor.items():
            new_dict[name] = clone(value)

        return new_dict
    elif isinstance(tensor, torch.Tensor):
        return tensor.clone()
    else:
        return np.copy(tensor)


def print_rank_0(message, file=None):
    """If distributed is initialized, print only on rank 0."""
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(message, file=file, flush=True)
    else:
        print(message, file=file, flush=True)


def cat_files(files, output, end="\n"):
    files = list(files)
    n_files = len(files)
    if n_files == 0:
        return

    is_fio = isinstance(output, io.IOBase)
    if not is_fio:
        out = open(output, mode="wb")
    else:
        out = output

    for i, path in enumerate(files):
        with open(path, mode="rb") as f:
            shutil.copyfileobj(f, out)
            if i < n_files - 1 and end:
                out.write(end.encode())

    if not is_fio:
        out.close()


def jloads(jline, mode="r"):
    if os.path.isfile(jline):
        data = []
        with open(jline, mode=mode, encoding="utf-8") as f:
            for line in f.readlines():
                data.append(json.loads(line))
        return data

    return json.loads(jline)


def jload(f, mode="r", object_pairs_hook=None):
    """Load a .json file into a dictionary."""
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    jdict = json.load(f, object_pairs_hook=object_pairs_hook)
    f.close()
    return jdict


def jdump(obj, f, mode="w", indent=4, default=str):
    """Dump a str or dictionary to a file in json format.
    Args:
        obj: An object to be written.
        f: A file handle or string path to the location on disk.
        mode: Mode for opening the file.
        indent: Indent for storing json dictionaries.
        default: A function to handle non-serializable entries; defaults to `str`.
    """
    is_fio = isinstance(f, io.IOBase)
    if not is_fio:
        folder = os.path.dirname(f)
        if folder != "":
            os.makedirs(folder, exist_ok=True)
        f = open(f, mode=mode)

    if isinstance(obj, (dict, list)):
        json.dump(obj, f, indent=indent, default=default)

    elif isinstance(obj, str):
        f.write(obj)
    else:
        raise ValueError(f"Unexpected type: {type(obj)}")

    if not is_fio:
        f.close()


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def get_cache_dir():
    home_dir = os.path.expanduser('~')
    cache_dir = os.path.join(home_dir, '.cache', "tgnn")
    os.makedirs(cache_dir, exist_ok=True)

    return cache_dir


def download_file(url, output, verbose=True):
    """ downloads remote_file to local_file if necessary """

    def bar_progress(current, total, width=80, desc="Downloading"):
        progress_message = f"{desc}: %d%% [%d / %d] bytes" % (current / total * 100, current, total)
        # Don't use print() as it will print in new line every time.
        sys.stdout.write("\r" + progress_message)
        sys.stdout.flush()

    if os.path.isfile(output):
        print(f"exist file: {output}")
        return output

    print(f"downloading {url} to {output}")
    name = wget.detect_filename(url)
    bar = partial(bar_progress, desc=f"downloading {name}") if verbose else None
    try:
        filename = wget.download(url, output, bar=bar)
    except:
        if os.path.isfile(output):
            os.remove(output)
        print(f"can not download: {url}")
        return None

    return filename


def set_file_timestamp(filename, access_time, modif_time=None):
    modif_time = access_time if modif_time is None else modif_time
    os.utime(filename, (access_time, modif_time))


def get_file_timestamp(filename):
    return os.path.getmtime(filename)
