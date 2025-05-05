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
import rootutils
import shutil
import sys
import torch
from typing import Any, Dict, List, Optional, Tuple

import hydra
import lightning as L
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig, OmegaConf, open_dict

from opencomplex.hydra_utils import (
    MultiDeviceLogger,
    extras,
    instantiate_callbacks,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
)
from sable.loss.loss import SableLoss
from sable.model.model import Sable
from sable.module.sable_module import SableModule

torch.set_float32_matmul_precision("high")
# os.environ["TORCH_CPP_LOG_LEVEL"]="INFO"
# os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"

# torch.backends.cuda.matmul.allow_tf32 = True
# torch.backends.cudnn.allow_tf32 = True

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/rootutils
# ------------------------------------------------------------------------------------ #



log = MultiDeviceLogger(__name__, rank_zero_only=True)


@task_wrapper
def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Trains the model. Can additionally evaluate on a testset, using best weights obtained during
    training.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: A DictConfig configuration composed by Hydra.
    :return: A tuple with metrics and dict with all instantiated objects.
    """
    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    # set lr scheduler
    epoch_total_samples, batch_size, total_epochs, devices = len(datamodule.train_dataset), cfg.data.batch_size, cfg.trainer.max_epochs, cfg.trainer.devices
    epoch_total_batchs = (epoch_total_samples + batch_size - 1) // batch_size
    epoch_total_steps = (epoch_total_batchs + devices - 1) // devices
    total_steps = epoch_total_steps * total_epochs
    OmegaConf.set_struct(cfg, True)
    with open_dict(cfg):
        cfg.module.scheduler.total_steps = total_steps

    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    log.info(f"Instantiating and initializing model <{cfg.module._target_}>")
    # polish the logic between train, fine-tune, and ckpt_path
    ckpt_path = cfg.get("ckpt_path")
    if not(cfg.get("fine-tune")) and ckpt_path: # the checkpoint is for the module itself
        log.info(f"Load module <{cfg.module._target_}> from checkpoint \"{ckpt_path}\"")
        loss: torch.nn.Module = hydra.utils.instantiate(cfg.module.loss)
        model: torch.nn.Module = hydra.utils.instantiate(cfg.module.model)
        module: LightningModule = hydra.utils.get_class(cfg.module._target_).load_from_checkpoint(ckpt_path, loss=loss, model=model)
    else: # train from scratch or the checkpoint is for fine-tuning from SableModule
        module: LightningModule = hydra.utils.instantiate(cfg.module) # initialize the downstream task model
        if cfg.get("parameter_init"):
            hydra.utils.instantiate(cfg.parameter_init, module=module)
        if ckpt_path: # load checkpoint of pre-training
            log.info(f"Load pre-trained model <sable.module.sable.Sable> from checkpoint \"{ckpt_path}\" for fine-tuning")
            loss: torch.nn.Module = SableLoss(config=cfg.module.sable_config.loss_config)
            model: torch.nn.Module = Sable(config=cfg.module.sable_config.model_config)
            sable_module = SableModule.load_from_checkpoint(ckpt_path, loss=loss, model=model) # load the pre-train model checkpoint
            if cfg.get("freeze"):
                sable_module.freeze()
            if cfg.module.model.config.sable.loss_distance_scalar < 0:
                delattr(sable_module.model, "dist_head")
            if cfg.module.model.config.sable.loss_residue_scalar < 0:
                delattr(sable_module.model, "lm_head")
            delattr(module.model, "sable")
            module.model.sable = sable_module.model # replace the model in downstream task with pre-train model
            del sable_module

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": module,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    if cfg.get("train"):
        log.info("Starting training!")
        trainer.fit(model=module, datamodule=datamodule)

    if trainer.checkpoint_callback.best_model_path:
        dir_path, best_ckpt_name = os.path.split(trainer.checkpoint_callback.best_model_path)
        shutil.copy2(trainer.checkpoint_callback.best_model_path, os.path.join(dir_path, "best_" + best_ckpt_name))

    train_metrics = trainer.callback_metrics

    if cfg.get("test"):
        log.info("Starting testing!")
        test_ckpt_path = trainer.checkpoint_callback.best_model_path
        if test_ckpt_path == "":
            log.warning("Best ckpt not found! Use ckpt_path in config")
            test_ckpt_path = ckpt_path if not(cfg.get("fine-tune")) else None # when it is fine-tune, the checkpoint is for pre-train model
            if test_ckpt_path is None or test_ckpt_path == "" or not os.path.exists(test_ckpt_path):
                test_ckpt_path = None
                log.warning("Ckpt path not found! Using current weights for testing...")

        trainer.test(model=module, datamodule=datamodule, ckpt_path=test_ckpt_path)
        log.info(f"Best ckpt path: {test_ckpt_path}")

    test_metrics = trainer.callback_metrics

    # merge train and test metrics
    metric_dict = {**train_metrics, **test_metrics}

    return metric_dict, object_dict


def construct_run_name(_root_: OmegaConf) -> str:
    OmegaConf.set_struct(_root_, False) # set the dataset in tags
    if _root_.task_name != "sable":
        if _root_.get("fine-tune"):
            _root_.tags["Fine-tune"] = 1
            if _root_.get("freeze"):
                _root_.tags["Freeze"] = 1
        elif _root_.get("train"):
            _root_.tags["From-Scratch"] = 1
        else:
            _root_.tags["Test"] = 1
    components = set()
    if _root_.dataset and _root_.dataset != "sable":
        _root_.tags[_root_.dataset] = 1
        components.add(_root_.dataset)
    for tag in _root_.tags.keys():
        if tag != _root_.task_name and tag != "sable":
            components.add(tag)
    return ("_".join(sorted(list(components), reverse=True)) + "_") if components else ""


OmegaConf.register_new_resolver("construct_run_name", construct_run_name)
@hydra.main(
    version_base="1.3", config_path="sable/config", config_name="sable.yaml"
)
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """

    # train the model
    train(cfg)


if __name__ == "__main__":

    if len(sys.argv) > 1 and any(map(lambda x: x.startswith("debug=") and x != "debug=none", sys.argv[1 : ])): # make verbose log for debug mode
        os.environ["HYDRA_FULL_ERROR"] = "1"
    main()

