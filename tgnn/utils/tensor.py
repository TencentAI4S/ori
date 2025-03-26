# Copyright (c) 2025, Tencent Inc. All rights reserved.
# Data: 2025/1/9 15:58
# Author: chenchenqin
from functools import partial
from typing import Union, List, Tuple

import numpy as np
import torch

from .type import is_tensor


# With tree_map, a poor man's JAX tree_map
def dict_map(fn, dic, leaf_type):
    new_dict = {}
    for k, v in dic.items():
        if type(v) is dict:
            new_dict[k] = dict_map(fn, v, leaf_type)
        else:
            new_dict[k] = tree_map(fn, v, leaf_type)

    return new_dict


tensor_dict_map = partial(dict_map, leaf_type=torch.Tensor)


def tree_map(fn, tree, leaf_type):
    if isinstance(tree, dict):
        return dict_map(fn, tree, leaf_type)
    elif isinstance(tree, list):
        return [tree_map(fn, x, leaf_type) for x in tree]
    elif isinstance(tree, tuple):
        return tuple([tree_map(fn, x, leaf_type) for x in tree])
    elif isinstance(tree, leaf_type):
        return fn(tree)
    else:
        raise ValueError(f"Not supported type: {type(tree)}")


tensor_tree_map = partial(tree_map, leaf_type=torch.Tensor)


def flatten_final_dims(t: torch.Tensor, num_dims: int):
    return t.reshape(t.shape[:-num_dims] + (-1,))


def permute_final_dims(tensor: torch.Tensor,
                       inds: Union[List, Tuple]):
    zero_index = -1 * len(inds)  # -2
    first_inds = list(range(len(tensor.shape[:zero_index])))
    orders = first_inds + [zero_index + i for i in inds]
    return tensor.permute(orders)


def masked_mean(mask, value, dim=None, keepdim=False, eps=1e-12):
    if mask is None:
        return value.mean(dim=dim, keepdim=keepdim)

    mask = mask.expand_as(value)
    return torch.sum(mask * value, dim=dim, keepdim=keepdim) / (eps + torch.sum(mask, dim=dim, keepdim=keepdim))


def dict_multimap(fn, dicts):
    first = dicts[0]
    new_dict = {}
    for k, v in first.items():
        all_v = [d[k] for d in dicts]
        if type(v) is dict:
            new_dict[k] = dict_multimap(fn, all_v)
        else:
            new_dict[k] = fn(all_v)

    return new_dict


def collate_dense_tensors(
        samples: List[torch.Tensor],
        pad_v: float = 0,
        max_shape: Tuple = None
) -> torch.Tensor:
    """collate batch tensor
    Takes a list of tensors with the following dimensions:
        [(d_11,       ...,           d_1K),
         (d_21,       ...,           d_2K),
         ...,
         (d_N1,       ...,           d_NK)]
    and stack + pads them into a single tensor of:
    (N, max_i=1,N { d_i1 }, ..., max_i=1,N {diK})
    """
    if len(samples) == 0:
        return torch.Tensor()

    # assert all tensor have same dim
    if len(set(x.dim() for x in samples)) != 1:
        raise RuntimeError(
            f"Samples has varying dimensions: {[x.dim() for x in samples]}"
        )

    (device,) = tuple(set(x.device for x in samples))  # assumes all on same device
    if max_shape is None:
        max_shape = [max(lst) for lst in zip(*[x.shape for x in samples])]

    result = torch.empty(
        len(samples), *max_shape, dtype=samples[0].dtype, device=device
    )
    result.fill_(pad_v)
    for i in range(len(samples)):
        result_i = result[i]
        t = samples[i]
        result_i[tuple(slice(0, k) for k in t.shape)] = t

    return result


def batched_gather(data, inds, dim=0, no_batch_dims=0):
    ranges = []
    # create batch dim index
    for i, s in enumerate(data.shape[:no_batch_dims]):
        r = torch.arange(s)
        r = r.view(*(*((1,) * i), -1, *((1,) * (len(inds.shape) - i - 1))))
        ranges.append(r)

    remaining_dims = [
        slice(None) for _ in range(len(data.shape) - no_batch_dims)
    ]
    remaining_dims[dim - no_batch_dims if dim >= 0 else dim] = inds
    ranges.extend(remaining_dims)
    return data[ranges]


def isin(element, test_elements, **kwargs):
    if is_tensor(element):
        if isinstance(test_elements, (list, tuple)):
            test_elements = torch.tensor(test_elements, dtype=element.dtype, device=element.device)
        return torch.isin(element, test_elements, **kwargs)
    else:
        return np.isin(element, test_elements, **kwargs)
