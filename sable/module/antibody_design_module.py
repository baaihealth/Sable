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
from omegaconf import DictConfig
import torch
from torch import Tensor, nn, optim
from torchmetrics.aggregation import MeanMetric, SumMetric

from sable.util.tensor_util import align_metric_device


class AntibodyDesignModule(LightningModule):
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

        super(AntibodyDesignModule, self).__init__()

        self.cdr_mask = model.cdr_mask
        self.loss = loss
        self.model = model
        self.metrics = [{ "count": SumMetric(), "hit": SumMetric() } for _ in range(3)]
        for chain, cdrs in self.cdr_mask.items():
            for cdr in cdrs:
                for i in range(1, 3):
                    self.metrics[i][chain + str(cdr) + "_count"] = SumMetric()
                    self.metrics[i][chain + str(cdr) + "_hit"] = SumMetric()
                self.metrics[2][chain + str(cdr) + "_RMSD"] = MeanMetric()
        self.save_hyperparameters(ignore=["cdr_mask", "loss", "metrics", "model"])

    def on_train_epoch_start(self) -> None:
        self.last_timestampe = datetime.datetime.utcnow()
        self.trainer.datamodule.train_dataset.reroll()

    def forward(self, x: Mapping[str, Any]) -> Tensor:
        """
        :param x: the input tensor dictionary for the forward procedure
        """

        return self.model(x)

    def training_step(self, batch: Mapping[str, Any]) -> Tensor:
        """
        :param batch: the input batch for the step
        """

        # Run the model
        output = self(batch)

        # Compute loss
        loss_breakdown = self.loss(output, batch)

        # Log the loss
        batch_size = batch["H_tgt_residue"].shape[0]
        self.log("train/AAR", loss_breakdown["hit"] / loss_breakdown["count"], on_step=True, on_epoch=False, logger=self.trainer.logger, sync_dist=False, batch_size=batch_size)
        self.log("train/distance_loss", loss_breakdown["distance_loss"], on_step=True, on_epoch=False, logger=self.trainer.logger, sync_dist=False, batch_size=batch_size)
        self.log("train/loss", loss_breakdown["loss"], on_step=True, on_epoch=False, logger=self.trainer.logger, sync_dist=False, batch_size=batch_size)
        self.log("train/residue_loss", loss_breakdown["residue_loss"], on_step=True, on_epoch=False, logger=self.trainer.logger, sync_dist=False, batch_size=batch_size)
        self.log("train/structure_loss", loss_breakdown["structure_loss"], on_step=True, on_epoch=False, logger=self.trainer.logger, sync_dist=False, batch_size=batch_size)

        self.log("AAR", loss_breakdown["hit"] / loss_breakdown["count"], prog_bar=True, on_step=True, logger=False, batch_size=batch_size)
        self.log("loss", loss_breakdown["loss"], prog_bar=True, on_step=True, logger=False, batch_size=batch_size)
        cur_lr=self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("lr", cur_lr, prog_bar=True, on_step=True, logger=False, batch_size=batch_size)

        align_metric_device(self.metrics[0], loss_breakdown)
        self.metrics[0]["count"].update(loss_breakdown["count"])
        self.metrics[0]["hit"].update(loss_breakdown["hit"])

        return loss_breakdown["loss"]

    def on_validation_epoch_start(self) -> None:
        total_count = self.metrics[0]["count"].compute()
        total_hit = self.metrics[0]["hit"].compute()
        ts_now = datetime.datetime.utcnow()
        rank_zero_info("\n%s | Training Epoch %d | Duration %s | AAR %.6f = %.0f / %.0f" % (str(ts_now), self.current_epoch, str(ts_now - self.last_timestampe), total_hit / total_count, total_hit, total_count))
        self.metrics[0]["count"].reset()
        self.metrics[0]["hit"].reset()
        self.last_timestampe = ts_now

    def validation_step(self, batch: Mapping[str, Any]) -> Tensor:
        """
        :param batch: the input batch for the step
        """

        # Run the model
        output = self(batch)

        # Compute loss
        loss_breakdown = self.loss(output, batch)

        # Log the loss
        self.log("valid/distance_loss", loss_breakdown["distance_loss"], on_step=False, on_epoch=True, logger=self.trainer.logger, sync_dist=True, batch_size=1)
        self.log("valid/loss", loss_breakdown["loss"], on_step=False, on_epoch=True, logger=self.trainer.logger, sync_dist=True, batch_size=1)
        self.log("valid/residue_loss", loss_breakdown["residue_loss"], on_step=False, on_epoch=True, logger=self.trainer.logger, sync_dist=True, batch_size=1)
        self.log("valid/structure_loss", loss_breakdown["structure_loss"], on_step=False, on_epoch=True, logger=self.trainer.logger, sync_dist=True, batch_size=1)

        align_metric_device(self.metrics[1], loss_breakdown)
        for chain, cdrs in self.cdr_mask.items():
            for cdr in cdrs:
                prefix = chain + str(cdr)
                self.metrics[1][prefix + "_count"].update(loss_breakdown[prefix + "_count"])
                self.metrics[1][prefix + "_hit"].update(loss_breakdown[prefix + "_hit"])
                self.log("valid/" + prefix + "_RMSD", loss_breakdown[prefix + "_RMSD"], on_step=False, on_epoch=True, logger=self.trainer.logger, sync_dist=True, batch_size=1)
        self.metrics[1]["count"].update(loss_breakdown["count"])
        self.metrics[1]["hit"].update(loss_breakdown["hit"])

        return loss_breakdown["loss"]

    def on_validation_epoch_end(self):
        for chain, cdrs in self.cdr_mask.items():
            for cdr in cdrs:
                prefix = chain + str(cdr)
                cdr_total_count = self.metrics[1][prefix + "_count"].compute()
                cdr_total_hit = self.metrics[1][prefix + "_hit"].compute()
                self.log("valid/" + prefix + "_AAR", cdr_total_hit / cdr_total_count, on_step=False, on_epoch=True, logger=self.trainer.logger, sync_dist=True, batch_size=1)
                self.metrics[1][prefix + "_count"].reset()
                self.metrics[1][prefix + "_hit"].reset()
        total_count = self.metrics[1]["count"].compute()
        total_hit = self.metrics[1]["hit"].compute()
        self.log("valid/AAR", total_hit / total_count, on_step=False, on_epoch=True, logger=self.trainer.logger, sync_dist=True, batch_size=1)
        ts_now = datetime.datetime.utcnow()
        rank_zero_info("\n%s | Validation Epoch %d | Duration %s | AAR %.6f = %.0f / %.0f" % (str(ts_now), self.current_epoch, str(ts_now - self.last_timestampe), total_hit / total_count, total_hit, total_count))
        self.metrics[1]["count"].reset()
        self.metrics[1]["hit"].reset()

    def summarize_test_results(self, test_idx: int):
        results = []
        for chain, cdrs in self.cdr_mask.items():
            for cdr in cdrs:
                prefix = chain + str(cdr)
                results.append((prefix, self.metrics[2][prefix + "_hit"].compute() / self.metrics[2][prefix + "_count"].compute(), self.metrics[2][prefix + "_RMSD"].compute()))
                self.metrics[2][prefix + "_RMSD"].reset()
                self.metrics[2][prefix + "_count"].reset()
                self.metrics[2][prefix + "_hit"].reset()
        results.sort(key=lambda x: x[0])
        cw = 14 # the column width
        rank_zero_info("\n====Testing Results for Antibody Design on \"%s\" with %d Cases====" % (
            os.path.normpath(self.trainer.datamodule.test_datasets[test_idx].data_path).split(os.sep)[-2],
            len(self.trainer.datamodule.test_datasets[test_idx]),
        ))
        rank_zero_info(
            ((" " * (cw + 1)) if len(results) > 1 else "") + "+" +
            ("-" * (len(results) * (cw + 1) - 1)) + "+" +
            ("-" * (len(results) * (cw + 1) - 1)) + "+"
        )
        rank_zero_info(
            ((" " * (cw + 1)) if len(results) > 1 else "") + "|" +
            (" " * ((len(results) * (cw + 1) - 3 - 1) // 2)) + "AAR" + (" " * ((len(results) * (cw + 1) - 3) // 2)) + "|" +
            (" " * ((len(results) * (cw + 1) - 4 - 1) // 2)) + "RMSD" + (" " * ((len(results) * (cw + 1) - 4) // 2)) + "|"
        )
        if len(results) > 1:
            rank_zero_info(
                "+" + ("-" * cw) + "+" +
                (("-" * cw) + "+") * (len(results) * 2)
            )
            rank_zero_info(
                "|" + (" " * ((cw - 3) // 2)) + "AAR" + (" " * ((cw - 3 + 1) // 2)) + "|" +
                "".join([((" " * ((cw - len(x[0])) // 2)) + x[0] + (" " * ((cw - len(x[0]) + 1) // 2)) + "|") for x in results]) +
                "".join([((" " * ((cw - len(x[0])) // 2)) + x[0] + (" " * ((cw - len(x[0]) + 1) // 2)) + "|") for x in results])
            )
        rank_zero_info(
            "+" + ((("-" * cw) + "+") if len(results) > 1 else "") +
            (("-" * cw) + "+") * (len(results) * 2)
        )
        value_format_str = " %%%d.6f |" % (cw - 2)
        rank_zero_info( # detailed metrics
            "|" + ((value_format_str % (self.metrics[2]["hit"].compute() / self.metrics[2]["count"].compute())) if len(results) > 1 else "") +
            "".join([(value_format_str % (x[1])) for x in results]) +
            "".join([(value_format_str % (x[2])) for x in results])
        )
        rank_zero_info(
            "+" + ((("-" * cw) + "+") if len(results) > 1 else "") +
            (("-" * cw) + "+") * (len(results) * 2)
        )
        self.metrics[2]["hit"].reset()
        self.metrics[2]["count"].reset()

    def test_step(self, batch: Mapping[str, Any], batch_idx: int, dataloader_idx: int=0) -> Tensor:
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
        for chain, cdrs in self.cdr_mask.items():
            for cdr in cdrs:
                prefix = chain + str(cdr)
                self.metrics[2][prefix + "_count"].update(loss_breakdown[prefix + "_count"])
                self.metrics[2][prefix + "_hit"].update(loss_breakdown[prefix + "_hit"])
                self.metrics[2][prefix + "_RMSD"].update(loss_breakdown[prefix + "_RMSD"])
        self.metrics[2]["count"].update(loss_breakdown["count"])
        self.metrics[2]["hit"].update(loss_breakdown["hit"])

        return loss_breakdown["loss"]

    def on_test_epoch_end(self):
        self.summarize_test_results(-1)

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

