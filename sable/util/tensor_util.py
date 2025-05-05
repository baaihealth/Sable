# Copyright 2025 Beijing Academy of Artificial Intelligence (BAAI)
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

from typing import Mapping

import torch
from torch import Tensor
import torch.nn.functional as F
from torchmetrics.metric import Metric


def get_activation_fn(activation: str):
    """
    Returns the activation function given `activation`

    :param activation: the name for the activtion function
    """

    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    elif activation == "tanh":
        return torch.tanh
    elif activation == "linear":
        return lambda x: x
    else:
        raise RuntimeError("activation function {} is not supported so far".format(activation))


def softmax_dropout(input: Tensor, dropout_prob: float, is_training: bool=True, mask: Tensor=None, bias: Tensor=None, inplace: bool=True) -> Tensor:
    """
    The procedure of applying softmax and then dropout

    :param input: the input tensor
    :param dropout_prob: probability of an element to be zeroed
    :param is_training: the dropout works only for training
    :param mask: the mask tensor on input
    :param bias: the bias tensor on input
    :param inplace: indicator for whether operation is done in-place, make a copy beforehand when it is not
    """

    input = input.contiguous()
    if not(inplace):
        # copy a input for non-inplace case
        input = input.clone()
    if mask is not None:
        input += mask
    if bias is not None:
        input += bias
    return F.dropout(F.softmax(input, dim=-1), p=dropout_prob, training=is_training)


def align_metric_device(metrics: Mapping[str, Metric], loss_breakdown: Mapping[str, Tensor]) -> None:
    """
    Unify the device settings in `metrics`, using the device information from `loss_breakdown`

    :param metrics: the metric elements to be aligned
    :param loss_breakdown: the loss breakdown where device information is borrowed from
    """

    # Note(chenxi): type in-consistency for non-hyperparameter metrics
    #     The `metrics` is designed to be a dict instead of seperate metrics directly, to avoid being part of the hyperparameters
    #     The consequence is that the DDP backend won't set its `device` corresponding and leading to the error when `update`: update with tensor in GPU to metric in CPU
    #     Setting all metrics in CPU (not GPU in case one wants to debug in CPU environment) will cause following error
    # https://github.com/Lightning-AI/pytorch-lightning/issues/18803
    target_device = next(iter(loss_breakdown.items()))[1].device
    if next(iter(metrics.items()))[1].device != target_device:
        for m in metrics.values():
            m.to(device=target_device)

