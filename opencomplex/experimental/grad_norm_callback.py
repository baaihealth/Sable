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

import torch
from lightning.pytorch.callbacks.callback import Callback

class GradNormCallback(Callback):
    """
    Logs the gradient norm.
    """

    def on_before_optimizer_step(self, trainer, model, optimizer):
        phase = "train" if model.training else "val"
        model.log(f"{phase}/grad_norm", gradient_norm(model))
        model.log(f"{phase}/grad_max", gradient_max(model))

        # turn on to find unused parameters
        # find_unused_parameter(model)

def find_unused_parameter(model):
    for name, param in model.named_parameters():
        if param.grad is None:
            print(name)

def gradient_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm


def gradient_max(model):
    max_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = torch.max(p.grad)
            max_norm = max(max_norm, param_norm)
    return float(max_norm)

