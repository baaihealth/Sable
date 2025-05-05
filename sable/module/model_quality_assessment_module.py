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

import os
from typing import Any, Mapping

from lightning import LightningModule
from lightning.pytorch.utilities.rank_zero import rank_zero_info
import torch
from torch import Tensor, nn, optim
from torchmetrics.metric import Metric
from torchmetrics.regression import KendallRankCorrCoef, MeanSquaredError, PearsonCorrCoef, SpearmanCorrCoef

from sable.util.tensor_util import align_metric_device


def update_metrics(metrics: Mapping[str, Metric], loss_breakdown: Mapping[str, Tensor]) -> None:
    align_metric_device(metrics, loss_breakdown)
    for k, m in metrics.items():
        tokens = k.lower().split()[ : -1]
        if tokens[0] == "average":
            m.update(loss_breakdown["pred_" + "_".join(tokens[1 : ])].mean().unsqueeze(dim=0), loss_breakdown["tgt_" + "_".join(tokens[1 : ])].mean().unsqueeze(dim=0))
        else:
            m.update(loss_breakdown["pred_" + "_".join(tokens)], loss_breakdown["tgt_" + "_".join(tokens)])


class ModelQualityAssessmentModule(LightningModule):
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

        super(ModelQualityAssessmentModule, self).__init__()

        self.loss = loss
        self.model = model
        self.metrics = [{
            "GDT TS RMSE": MeanSquaredError(squared=False),
            "GDT TS Pearson": PearsonCorrCoef(),
            "GDT TS Spearman": SpearmanCorrCoef(),
            "GDT TS Kendall": KendallRankCorrCoef(),
            "Average LDDT RMSE": MeanSquaredError(squared=False),
            "Average LDDT Pearson": PearsonCorrCoef(),
            "Average LDDT Spearman": SpearmanCorrCoef(),
            "Average LDDT Kendall": KendallRankCorrCoef(),
            "LDDT Pearson": PearsonCorrCoef(),
            "LDDT Spearman": SpearmanCorrCoef(),
        } for _ in range(2)]
        self.save_hyperparameters(ignore=["loss", "metrics", "model"])

    def on_train_epoch_start(self) -> None:
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
        self.log("train/loss", loss_breakdown["loss"], on_step=True, on_epoch=False, logger=self.trainer.logger, sync_dist=False)
        self.log("train/gdt_ts_loss", loss_breakdown["gdt_ts_loss"], on_step=True, on_epoch=False, logger=self.trainer.logger, sync_dist=False)
        self.log("train/lddt_loss", loss_breakdown["lddt_loss"], on_step=True, on_epoch=False, logger=self.trainer.logger, sync_dist=False)

        self.log("loss", loss_breakdown["loss"], prog_bar=True, on_step=True, logger=False)
        self.log("gdt_ts_loss", loss_breakdown["gdt_ts_loss"], prog_bar=True, on_step=True, logger=False)
        self.log("lddt_loss", loss_breakdown["lddt_loss"], prog_bar=True, on_step=True, logger=False)
        cur_lr=self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("lr", cur_lr, prog_bar=True, on_step=True, logger=False)

        return loss_breakdown["loss"]

    def validation_step(self, batch: Mapping[str, Tensor]) -> Tensor:
        """
        :param batch: the input batch for the step
        """

        # Run the model
        output = self(batch)

        # Compute loss
        loss_breakdown = self.loss(output, batch)

        # Log the loss
        self.log("valid/loss", loss_breakdown["loss"], on_step=False, on_epoch=True, logger=self.trainer.logger, sync_dist=True)
        self.log("valid/gdt_ts_loss", loss_breakdown["gdt_ts_loss"], on_step=False, on_epoch=True, logger=self.trainer.logger, sync_dist=True)
        self.log("valid/lddt_loss", loss_breakdown["lddt_loss"], on_step=False, on_epoch=True, logger=self.trainer.logger, sync_dist=True)

        update_metrics(self.metrics[0], loss_breakdown)

        return loss_breakdown["loss"]

    def on_validation_epoch_end(self):
        for (k, v) in self.metrics[0].items():
            self.log("valid/" + "_".join(k.lower().split()), v.compute(), on_step=False, on_epoch=True, logger=self.trainer.logger, sync_dist=True)
            v.reset()

    def summarize_test_results(self, test_idx: int):
        """
        :param test_idx: index for a testset, follows the python way of indexing directly that -1 for the last
        """

        output_columns = [("GDT TS", ["RMSE", "Pearson", "Spearman", "Kendall"]), ("Average LDDT", ["RMSE", "Pearson", "Spearman", "Kendall"]), ("LDDT", ["Pearson", "Spearman"])]
        cw = 14 # the column width
        rank_zero_info("\n====Testing Results for Model Quality Assessment on \"%s\" with %d Cases====" % (
            os.path.normpath(self.trainer.datamodule.test_datasets[test_idx].data_path).split(os.sep)[-2],
            len(self.trainer.datamodule.test_datasets[test_idx]),
        ))
        rank_zero_info("+" + "".join([(("-" * (len(c[1]) * (cw + 1) - 1)) + "+") for c in output_columns]))
        rank_zero_info("|" + "".join([((" " * ((len(c[1]) * (cw + 1) - len(c[0]) - 1) // 2)) + c[0] + (" " * ((len(c[1]) * (cw + 1) - len(c[0])) // 2)) + "|") for c in output_columns]))
        rank_zero_info("+" + "".join([(("-" * (len(c[1]) * (cw + 1) - 1)) + "+") for c in output_columns]))
        rank_zero_info("|" + "".join([(" " * ((cw - len(m)) // 2) + m + " " * ((cw - len(m) + 1) // 2) + "|") for c in output_columns for m in c[1]]))
        rank_zero_info("+" + "".join([(("-" * (len(c[1]) * (cw + 1) - 1)) + "+") for c in output_columns]))
        value_format_str = " %%%d.6f |" % (cw - 2)
        rank_zero_info("|" + "".join([(value_format_str % (self.metrics[1][c[0] + " " + m].compute())) for c in output_columns for m in c[1]]))
        rank_zero_info("+" + "".join([(("-" * (len(c[1]) * (cw + 1) - 1)) + "+") for c in output_columns]))
        for m in self.metrics[1].values():
            m.reset()

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
        update_metrics(self.metrics[1], loss_breakdown)

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

