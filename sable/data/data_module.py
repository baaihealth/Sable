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
import copy
import os
import traceback
from typing import List, Optional

from lightning import LightningDataModule
import lmdb
import pickle
from torch.utils.data import DataLoader
import torch

from sable.data.data_factory import SableDataFactory


class SableDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_factory: SableDataFactory,
        data_path: str,
        epoch_len: int=-1,
        persist_in_ram: bool=False,
        **kwargs,
    ):
        """
        :param data_factory: Works for extracting features and collating data samples in a batch
        :param data_path: The path to the LMDB file for the dataset
        :param epoch_len: The epoch length, which does not provide constrain for negative value
        :param persist_in_ram: Argument indicating whether to avoid accessing disk for each `__getitem__` call
        """

        super(SableDataset, self).__init__()

        self.current_epoch = -1
        self.data_factory = data_factory
        if os.path.isfile(data_path):
            self.data_path = data_path
        else:
            self.data_path = ""
            return
        self.persist_in_ram = persist_in_ram
        if self.persist_in_ram:
            environment = lmdb.open(data_path, subdir=False, lock=False)
            self.epoch_len = environment.stat()["entries"]
            if epoch_len > 0 and epoch_len < self.epoch_len: # mostly for debugging
                self.epoch_len = epoch_len
            transaction = environment.begin()
            self.datapoints = []
            for i in range(self.epoch_len):
                self.datapoints.append(pickle.loads(transaction.get(str(i).encode("ascii"))))
            environment.close()
        else:
            self.transaction = lmdb.open(data_path, subdir=False, lock=False).begin()
            self.epoch_len = self.transaction.stat()["entries"]
            if epoch_len > 0 and epoch_len < self.epoch_len: # mostly for debugging
                self.epoch_len = epoch_len

    def __getitem__(self, idx: int):
        """
        :param idx: the index for datapoint to fetch
        """

        try:
            if self.persist_in_ram:
                data = copy.deepcopy(self.datapoints[idx])
            else:
                data = pickle.loads(self.transaction.get(str(int(idx)).encode("ascii")))

            features = self.data_factory.featurizer(data, self.current_epoch, idx)

            return features
        except Exception as e:
            # Manually handle exception to fix lightning bug
            traceback.print_exception(None, e, e.__traceback__)
            raise e

    def __len__(self):
        return self.epoch_len

    def reroll(self) -> None:
        """Function to increase the `current_epoch` for entering the next epoch"""

        self.current_epoch += 1


class SableDataModule(LightningDataModule):
    def __init__(
        self,
        batch_size: int=1,
        num_workers: int=0,
        seed: Optional[int]=None,
        train_dataset: Optional[SableDataset]=None,
        eval_dataset: Optional[SableDataset]=None,
        test_datasets: Optional[List[SableDataset]]=None,
        **kwargs,
    ):
        """
        :param batch_size: Majorly works for training mode, it is 1 for other modes
        :param num_workers: Number of extra workers for data loaders
        :param seed: The random seed for reproducible runs
        :param train_dataset: The dataset for training, along with its featurizer
        :param eval_dataset: The dataset for validation, along with its featurizer
        :param test_datasets: The list of datasets for testing, along with its featurizer
        """

        super(SableDataModule, self).__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.generator = torch.Generator()
        if not(seed is None):
            self.generator.manual_seed(seed)
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        if self.eval_dataset is not None:
            self.val_dataloader = self.val_dataloader_override_function # get rid of the warning that with val_dataloader but without validation_step
        self.test_datasets = list(filter(lambda x: x.data_path, test_datasets)) if test_datasets else test_datasets

    def _gen_dataloader(self, stage: str) -> DataLoader:
        """
        Data loader router according to the `stage` argument

        :param stage: The stage comes from ["train", "eval", "test"]
        """

        dataset = None
        if stage == "train":
            dataset = self.train_dataset
        elif stage == "eval":
            dataset = self.eval_dataset
        elif stage == "test":
            dataset = self.test_datasets
        else:
            raise ValueError("Invalid stage")
        if dataset is None:
            return None
        if isinstance(dataset, SableDataset):
            dl = DataLoader(
                dataset,
                shuffle=(stage == "train"),
                generator=self.generator,
                batch_size=self.batch_size if (stage == "train") else 1,
                num_workers=self.num_workers,
                collate_fn=dataset.data_factory.batch_collator,
            )
        else:
            dl = [DataLoader(
                d,
                shuffle=False,
                batch_size=1,
                num_workers=self.num_workers,
                collate_fn=d.data_factory.batch_collator,
            ) for d in dataset]

        return dl

    def train_dataloader(self) -> DataLoader:
        """Obtain the dataloader for training"""

        return self._gen_dataloader("train")

    def val_dataloader_override_function(self) -> DataLoader:
        """Obtain the dataloader for validation"""

        return self._gen_dataloader("eval")

    def test_dataloader(self) -> DataLoader:
        """Obtain the dataloader for testing"""

        return self._gen_dataloader("test")

