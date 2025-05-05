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

import math

from lightning import LightningModule
from torch import nn


def variance_init(module: LightningModule) -> None:
    """
    The initialization method that apply random values in weights sampled from normal distribution

    :param module: The `LightningModule` to be initialized, now it is sure that `named_parameters` exists
    :param std: Initialize the weights sampling from the normal distribution, and `std` is the standard deviation
    """

    def normal_(data):
        data.copy_(data.cpu().normal_(mean=0.0, std=0.02).to(data.device))

    for m in module.modules():
        if isinstance(m, nn.Linear):
            normal_(m.weight.data)
            if not(m.bias is None):
                m.bias.data.zero_()
        elif isinstance(m, nn.Embedding):
            normal_(m.weight.data)
            if not(m.padding_idx is None):
                m.weight.data[m.padding_idx].zero_()


def kaiming_init(module: LightningModule) -> None:
    """
    The initialization method by Kaiming He from https://arxiv.org/abs/1502.01852

    :param module: The `LightningModule` to be initialized, now it is sure that `named_parameters` exists
    """

    for name, param in module.named_parameters():
        if ".sm." in name: # StructureModuleProtein has its own initialization way
            continue
        if name.endswith(".bias"):
            param.data.fill_(0)
        elif len(param.shape) < 2: # special case for LayerNorm
            continue
        elif ".layers.0" in name:  # The first layer does not have ReLU/GeLU applied on its input
            param.data.normal_(0, 1 / math.sqrt(param.shape[1]))
        else: # It is 0.5 for ReLU and 0.645 for GeLU
            param.data.normal_(0, 1 / math.sqrt(0.645 * param.shape[1]))

