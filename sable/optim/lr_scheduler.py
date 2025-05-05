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

from typing import Any, Mapping

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


class SableLRScheduler(LRScheduler):
    def __init__(self,
        optimizer: Optimizer,
        max_lr: float,
        total_steps: int,
        buffer: int=1,
        warmup_ratio: float=None,
        estimated_total_steps: int=None,
        warmup_steps: int=None,
        last_step: int = -1
    ):
        """
        Makes learning rate increase linearly in warmup stage, then decrease linearly during descending stage, and finally keep constant
        :param optimizer: The corresponding optimizer to pass learning rate
        :param max_lr: The maximum value of learning rate between the warmup and descending stage
        :param total_steps: The number of steps for the whole training
        :param buffer: Used to make the learning value non-zero, that move the number of step with zero learning rate with this number afterwards
        :param warmup_ratio: The ratio in `total_steps` as the warmup stage, so the warmup steps should be `warmup_ratio * total_steps`
        :param estimated_total_steps: Come together with `warmup_steps` and mutually exclusive to `warmup_ratio`, which define the warmup steps in a different way
        :param warmup_steps: Come together with `estimated_total_steps` and mutually exclusive to `warmup_ratio`, which define the warmup steps in a different way
        :param last_step: Record the steps status, it is useful for both calculating learning rate and recovering
        """

        self.optimizer = optimizer

        assert max_lr > 0 and total_steps > 0
        self.max_lr = max_lr
        buffer = max(buffer, 1) # buffer is positive

        if not(warmup_steps is None): # decided by `warmup_steps` and `estimated_total_steps` together
            self.warmup_steps = min(warmup_steps, total_steps)
            assert self.warmup_steps < estimated_total_steps
            self.plateaus_steps = min(estimated_total_steps, total_steps)
            self.zero_lr_steps = estimated_total_steps + buffer # the steps that learning rate reaches zero
        else: # decided by `warmup_ratio`
            assert 0 <= warmup_ratio <= 1
            self.warmup_steps = int(total_steps * warmup_ratio)
            self.plateaus_steps = total_steps
            self.zero_lr_steps = total_steps + buffer
        self.begin_lr = max_lr / self.warmup_steps

        self.last_step = last_step

        if self.last_step == -1:
            self.step()

    def get_lr(self) -> float:
        return self.optimizer.param_groups[0]['lr']

    def step(self) -> None:
        current_step = self.last_step + 1
        self.last_step = current_step

        if current_step <= self.warmup_steps: # in warmup stage, increase linearly
            lr = self.begin_lr + 1.0 * current_step * (self.max_lr - self.begin_lr) / self.warmup_steps
        elif current_step >= self.plateaus_steps: # after descending stage, keep the same learning rate
            lr = self.max_lr * (self.zero_lr_steps - self.plateaus_steps) / (self.zero_lr_steps - self.warmup_steps)
        else: # descending stage
            lr = self.max_lr * (self.zero_lr_steps - current_step) / (self.zero_lr_steps - self.warmup_steps)

        self.optimizer.param_groups[0]['lr'] = lr

