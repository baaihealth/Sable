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

import datetime
import os
from typing import Any, Mapping

from lightning import LightningModule
from lightning.pytorch.utilities.rank_zero import rank_zero_info
import torch
from torch import Tensor, nn, optim
from torchmetrics.aggregation import MeanMetric, SumMetric

from sable.util.tensor_util import align_metric_device


class ProteinDesignModule(LightningModule):
    def __init__(
        self,
        loss: nn.Module,
        model: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: optim.lr_scheduler,
        *args,
        **kwargs,
    ):
        """
        :param loss: The loss for feedback
        :param model: The model to train or load
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        """

        super(ProteinDesignModule, self).__init__()

        self.loss = loss
        self.model = model
        self.metrics = [ { "count": SumMetric(), "hit": SumMetric(), "PPL": MeanMetric() } for _ in range(3) ]
        self.save_hyperparameters(ignore=["loss", "metrics", "model"])

    def on_train_epoch_start(self) -> None:
        self.last_timestampe = datetime.datetime.utcnow()
        self.trainer.datamodule.train_dataset.reroll()

    def forward(self, x: Mapping[str, Tensor]) -> Tensor:
        """
        :param x: the input tensor dictionary for the forward procedure
        """

        return self.model(x)

    def training_step(self, batch: Mapping[str, Tensor]) -> Tensor:
        """
        :param batch: the input batch for the step
        """

        # Run the model
        output = self(batch)

        # Compute loss
        loss_breakdown = self.loss(output, batch)

        # Log the loss
        self.log("train/AAR", loss_breakdown["hit"] / loss_breakdown["count"], on_step=True, on_epoch=False, logger=self.trainer.logger, sync_dist=False)
        self.log("train/loss", loss_breakdown["loss"], on_step=True, on_epoch=False, logger=self.trainer.logger, sync_dist=False)

        self.log("AAR", loss_breakdown["hit"] / loss_breakdown["count"], prog_bar=True, on_step=True, logger=False)
        self.log("loss", loss_breakdown["loss"], prog_bar=True, on_step=True, logger=False)
        cur_lr=self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("lr", cur_lr, prog_bar=True, on_step=True, logger=False)

        align_metric_device(self.metrics[0], loss_breakdown)
        self.metrics[0]["PPL"].update(loss_breakdown["loss"].exp())
        self.metrics[0]["count"].update(loss_breakdown["count"])
        self.metrics[0]["hit"].update(loss_breakdown["hit"])

        return loss_breakdown["loss"]

    def on_validation_epoch_start(self) -> None:
        total_count = self.metrics[0]["count"].compute()
        total_hit = self.metrics[0]["hit"].compute()
        ts_now = datetime.datetime.utcnow()
        rank_zero_info("\n%s | Training Epoch %d | Duration %s | AAR %.6f = %.0f / %.0f | PPL %.6f" % (str(ts_now), self.current_epoch, str(ts_now - self.last_timestampe), total_hit / total_count, total_hit, total_count, self.metrics[0]["PPL"].compute()))
        self.metrics[0]["PPL"].reset()
        self.metrics[0]["count"].reset()
        self.metrics[0]["hit"].reset()
        self.last_timestampe = ts_now

    def validation_step(self, batch: Mapping[str, Tensor]) -> Tensor:
        """
        :param batch: the input batch for the step
        """

        # Run the model
        output = self(batch)

        # Compute loss
        loss_breakdown = self.loss(output, batch)

        # Log the loss
        self.log("valid/PPL", loss_breakdown["loss"].exp(), on_step=False, on_epoch=True, logger=self.trainer.logger, sync_dist=True)
        self.log("valid/loss", loss_breakdown["loss"], on_step=False, on_epoch=True, logger=self.trainer.logger, sync_dist=True)

        align_metric_device(self.metrics[1], loss_breakdown)
        self.metrics[1]["PPL"].update(loss_breakdown["loss"].exp())
        self.metrics[1]["count"].update(loss_breakdown["count"])
        self.metrics[1]["hit"].update(loss_breakdown["hit"])

        return loss_breakdown["loss"]

    def on_validation_epoch_end(self):
        total_count = self.metrics[1]["count"].compute()
        total_hit = self.metrics[1]["hit"].compute()
        self.log("valid/AAR", total_hit / total_count, on_step=False, on_epoch=True, logger=self.trainer.logger, sync_dist=True)
        ts_now = datetime.datetime.utcnow()
        rank_zero_info("\n%s | Validation Epoch %d | Duration %s | AAR %.6f = %.0f / %.0f | PPL %.6f" % (str(ts_now), self.current_epoch, str(ts_now - self.last_timestampe), total_hit / total_count, total_hit, total_count, self.metrics[1]["PPL"].compute()))
        self.metrics[1]["PPL"].reset()
        self.metrics[1]["count"].reset()
        self.metrics[1]["hit"].reset()

    def summarize_test_results(self, test_idx: int):
        """
        :param test_idx: index for a testset, follows the python way of indexing directly that -1 for the last
        """

        total_count = self.metrics[2]["count"].compute()
        total_hit = self.metrics[2]["hit"].compute()
        cw = 14 # the column width
        rank_zero_info("\n====Testing Results for Protein Design on \"%s\" with %d Cases====" % (
            os.path.normpath(self.trainer.datamodule.test_datasets[test_idx].data_path).split(os.sep)[-2],
            len(self.trainer.datamodule.test_datasets[test_idx]),
        ))
        rank_zero_info("+" + (("-" * cw) + "+") * 2)
        rank_zero_info("|" + (" " * ((cw - 3) // 2)) + "AAR" + (" " * ((cw - 3 + 1) // 2)) + "|" + (" " * ((cw - 3) // 2)) + "PPL" + (" " * ((cw - 3 + 1) // 2)) + "|") # detailed header
        rank_zero_info("+" + (("-" * cw) + "+") * 2)
        value_format_str = " %%%d.6f |" % (cw - 2)
        rank_zero_info(("|" + value_format_str * 2) % (total_hit / total_count, self.metrics[2]["PPL"].compute())) # detailed metrics
        rank_zero_info("+" + (("-" * cw) + "+") * 2)
        self.metrics[2]["PPL"].reset()
        self.metrics[2]["count"].reset()
        self.metrics[2]["hit"].reset()

    def test_step(self, batch: Mapping[str, Tensor], batch_idx: int, dataloader_idx: int=0) -> Tensor:
        """
        :param batch: the input batch for the step
        :param batch_idx: batch index value, which is not used
        :param dataloader_idx: dataloader index which is used to distinguish different test sets
        """

        if hasattr(self, "current_test_idx") and dataloader_idx != self.current_test_idx:
            if not(hasattr(self, "test_progress_bar_id")) or (self.test_progress_bar_id is None):
                self.test_progress_bar_id = dataloader_idx
            self.summarize_test_results(self.current_test_idx)
            self.current_test_idx = dataloader_idx
        else:
            if not(hasattr(self, "test_progress_bar_id")) or (self.test_progress_bar_id is None):
                self.test_progress_bar_id = dataloader_idx
            self.current_test_idx = dataloader_idx

        # Run the model
        output = self(batch)

        # Compute loss
        loss_breakdown = self.loss(output, batch)

        # Log the loss
        align_metric_device(self.metrics[2], loss_breakdown)
        self.metrics[2]["PPL"].update(loss_breakdown["loss"].exp())
        self.metrics[2]["count"].update(loss_breakdown["count"])
        self.metrics[2]["hit"].update(loss_breakdown["hit"])

        return loss_breakdown["loss"]

    def on_test_epoch_end(self):
        self.summarize_test_results(-1) # summarize the last test

    def configure_optimizers(self) -> Mapping[str, Any]:
        """
        Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.
        Examples: https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers
        """

        params = filter(lambda p: p.requires_grad, self.parameters())
        optimizer = self.hparams.optimizer(params=params)
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}

