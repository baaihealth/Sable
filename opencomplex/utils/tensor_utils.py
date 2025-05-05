# Copyright 2025 Beijing Academy of Artificial Intelligence (BAAI)
# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List

import torch

import numpy as np


def permute_final_dims(tensor: torch.Tensor, inds: List[int]):
    zero_index = tensor.dim() - len(inds)
    first_inds = list(range(zero_index))
    return tensor.permute(first_inds + [zero_index + i for i in inds])


def flatten_final_dims(t: torch.Tensor, no_dims: int):
    return torch.flatten(t, start_dim=-no_dims, end_dim=-1)


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


def batched_gather(data, inds, dim=0, no_batch_dims=0):
    ranges = []
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



def padcat(tensors, axis=0):
    tensors = [t for t in tensors if t is not None and t.shape[axis] > 0]
    if len(tensors) == 1:
        return tensors[0]

    ndim = tensors[0].ndim
    if axis < 0:
        axis += ndim

    axis_max_len = [
        max(t.shape[i] for t in tensors)
        for i in range(ndim)
    ]

    is_np = False
    if not isinstance(tensors[0], torch.Tensor):
        if tensors[0].dtype == np.object_:
            return np.concatenate(tensors, axis=axis)

        is_np = True
        tensors = [torch.tensor(t) for t in tensors]
    
    for i, t in enumerate(tensors):
        pads = [0 for _ in range(ndim * 2)]
        for j in range(0, ndim):
            if j != axis:
                pads[(ndim - j - 1) * 2 + 1] = axis_max_len[j] - t.shape[j]

        if any(pads):
            tensors[i] = torch.nn.functional.pad(t, tuple(pads))

    ret = torch.cat(tensors, axis=axis)
    if is_np:
        return ret.numpy()

    return ret


def map_padcat(a, b, axis=0):
    for i, j in zip(a, b):
        yield padcat([i, j], axis)


def padto(t, shape):
    ndim = len(shape)
    pads = [0 for _ in range(ndim * 2)]
    for j in range(0, ndim):
        pads[(ndim - j - 1) * 2 + 1] = shape[j] - t.shape[j]
    if any(pads):
        t = torch.nn.functional.pad(t, tuple(pads))
    return t

