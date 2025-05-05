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

from functools import partial
from typing import Any, Mapping, Sequence, Union

from omegaconf import DictConfig
from torch import Tensor
import torch

from sable.np import residue_constant as rc
from sable.data import data_transform as dt


class SableBatchCollactor:
    """Router for providing different functions to collate features in different types"""

    def __init__(self, atom_level: int):
        """
        :param atom_level: Number of atom coordinates provided in a protein residue
        """

        self.coordinate_collate_fn = dt.feature_padding(atom_level=atom_level)
        self.distance_collate_fn = dt.feature_padding_2d(atom_level=atom_level)
        self.lddt_collate_fn = dt.feature_padding()
        self.residue_collate_fn = dt.feature_padding(pad_token=rc.PAD)
        self.default_collate_fn = partial(torch.stack, dim=0)

    def __call__(self, sample_features: Sequence[Mapping[str, Any]]) -> Mapping[str, Union[Tensor, Sequence[Mapping[str, Any]]]]:
        """
        :param sample_features: The corresponding feature set for samples in a batch, each is a `dict` with same key set
        """

        ret = dict()
        for k, v in sample_features[0].items():
            tensor_list = [f[k] for f in sample_features] # collect same features from samples
            if "coordinate" in k:
                ret[k] = self.coordinate_collate_fn(tensor_list)
            elif "distance" in k:
                ret[k] = self.distance_collate_fn(tensor_list)
            elif "lddt" in k:
                ret[k] = self.lddt_collate_fn(tensor_list)
            elif "residue" in k:
                ret[k] = self.residue_collate_fn(tensor_list)
            elif "pdb" in k: # for PDB data, and original cdr does nothing
                ret[k] = tensor_list
            else:
                ret[k] = self.default_collate_fn(tensor_list)
        return ret


class SableFeaturizer():
    def __init__(self, config: DictConfig):
        """
        :param config: The configuration for the featurizer
        """

        self.config = config

    def __call__(self, bio_complex: Mapping[str, Tensor], epoch_id: int, idx: int) -> Mapping[str, Tensor]:
        """
        :param bio_complex: The original data to pass the feature extracting pipeline
        :param epoch_id: Current epoch id, starting from 0
        :param idx: The index for the datapoint, it is used for identifying datapoint and randomizing along with `epoch_id` as part of the seed
        """

        transforms = []

        transforms.append(dt.sable_data_preprocessing)

        if self.config.get("crop"):
            transforms.append(dt.crop_to_size(
                self.config.atom_level,
                self.config.crop.crop_size,
                self.config.crop.spatial_crop_ratio,
                seed=int(hash((self.config.get("seed", 42), (epoch_id, idx))) % 999983),
            ))

        transforms.append(dt.normalize_coordinate(key="int_coordinate"))
        transforms.append(dt.make_distance(src_key="int_coordinate", tgt_key="tgt_distance")) # prepare the `tgt_distance`
        transforms.append(dt.prepend_and_append_2d(key="tgt_distance", token=0.0, width=self.config.atom_level))

        transforms.append(
            dt.mask_protein(
                self.config.atom_level,
                self.config.mask.fake_p,
                self.config.mask.kept_p,
                seed=int(hash((self.config.get("seed", 42), (epoch_id, idx + 1))) % 999983),
            )
        )

        transforms.append(dt.prepend_and_append(src_key="src_residue", pre_token=rc.BOS, app_token=rc.EOS, repeat_mode=1, tgt_key="src_residue")) # prepend and append (bos and eos) to prepare `src_residue`
        transforms.append(dt.make_distance(src_key="src_coordinate", tgt_key="src_distance")) # prepare the `src_distance`
        transforms.append(dt.prepend_and_append_2d(key="src_distance", token=0.0, width=self.config.atom_level))
        transforms.append(dt.prepend_and_append(src_key="src_coordinate", pre_token=0.0, app_token=0.0, repeat_mode=(self.config.atom_level, 1), tgt_key="src_coordinate")) # prepare coordinates as `src_coordinate`
        transforms.append(dt.prepend_and_append(src_key="tgt_residue", pre_token=rc.PAD, app_token=rc.PAD, repeat_mode=1, tgt_key="tgt_residue")) # prepend and append (pad) to prepare `tgt_residue`

        transforms.append(dt.sable_feature_postprocessing) # remove the unnecessary fields

        transforms.append(dt.cast_to_64bit_ints)
        transforms.append(dt.cast_to_32bit_floats)

        for f in transforms:
            bio_complex = f(bio_complex)

        return bio_complex


class SableDataFactory():
    def __init__(self, config: DictConfig):
        """
        :param config: The configuration for the data factory, majorily for the featurizer
        """

        self.featurizer = SableFeaturizer(config)
        self.batch_collator = SableBatchCollactor(config.atom_level)


class ProteinDesignFeaturizer():
    def __init__(self, config: DictConfig):
        """
        :param config: The configuration for the featurizer
        """

        self.config = config

    def __call__(self, bio_complex: Mapping[str, Tensor], epoch_id: int, idx: int) -> Mapping[str, Tensor]:
        """
        :param bio_complex: The original data to pass the feature extracting pipeline
        :param epoch_id: Current epoch id, starting from 0
        :param idx: The index for the datapoint, it is used for identifying datapoint and randomizing along with `epoch_id` as part of the seed
        """

        transforms = []

        transforms.append(dt.protein_design_data_preprocessing)

        if self.config.get("crop"):
            transforms.append(dt.crop_to_size(
                self.config.atom_level,
                self.config.crop.crop_size,
                self.config.crop.spatial_crop_ratio,
                seed=int(hash((self.config.get("seed", 42), (epoch_id, idx + 2))) % 999983),
                extra=[ "tgt_residue" ],
            ))

        transforms.append(dt.normalize_coordinate(key="int_coordinate"))
        transforms.append(dt.prepend_and_append(src_key="int_residue", pre_token=rc.BOS, app_token=rc.EOS, repeat_mode=1, tgt_key="src_residue")) # prepend and append (bos and eos) to prepare `src_residue`
        transforms.append(dt.make_distance(src_key="int_coordinate", tgt_key="src_distance")) # prepare the `src_distance`
        transforms.append(dt.prepend_and_append_2d(key="src_distance", token=0.0, width=self.config.atom_level))
        transforms.append(dt.prepend_and_append(src_key="int_coordinate", pre_token=0.0, app_token=0.0, repeat_mode=(self.config.atom_level, 1), tgt_key="src_coordinate")) # prepare coordinates as `src_coordinate`
        transforms.append(dt.prepend_and_append(src_key="tgt_residue", pre_token=rc.BOS, app_token=rc.EOS, repeat_mode=1, tgt_key="tgt_residue")) # prepend and append (pad) to prepare `tgt_residue`

        transforms.append(dt.protein_design_feature_postprocessing) # remove the unnecessary fields

        transforms.append(dt.cast_to_64bit_ints)
        transforms.append(dt.cast_to_32bit_floats)

        for f in transforms:
            bio_complex = f(bio_complex)

        return bio_complex


class ProteinDesignDataFactory():
    def __init__(self, config: DictConfig):
        """
        :param config: The configuration for the data factory, majorily for the featurizer
        """

        self.featurizer = ProteinDesignFeaturizer(config)
        self.batch_collator = SableBatchCollactor(config.atom_level)


class AntibodyDesignFeaturizer():
    def __init__(self, config):
        """
        :param config: The configuration for the featurizer
        """

        self.config = config

    def __call__(self, bio_complex: Mapping[str, Any], epoch_id: int, idx: int) -> Union[Mapping[str, Tensor], Mapping[str, Mapping[str, Any]]]:
        """
        :param bio_complex: The original data to pass the feature extracting pipeline
        :param epoch_id: Current epoch id, starting from 0
        :param idx: The index for the datapoint, it is used for identifying datapoint and randomizing along with `epoch_id` as part of the seed
        """

        transforms = []

        transforms.append(dt.antibody_design_data_preprocessing(self.config.cdr_mask))

        if self.config.get("crop") and ("Ag" in self.config.cdr_mask):
            transforms.append(dt.crop_to_size(
                self.config.atom_level,
                self.config.crop.crop_size,
                self.config.crop.spatial_crop_ratio,
                seed=int(hash((self.config.get("seed", 42), (epoch_id, idx + 3))) % 999983),
                prefix="Ag_",
                extra=[
                    "Ag_pdb.aatype",
                    "Ag_pdb.asym_id",
                    "Ag_pdb.atom_mask",
                    "Ag_pdb.atom_positions",
                    "Ag_pdb.b_factors",
                    "Ag_pdb.chain_index",
                    "Ag_pdb.residue_index",
                ]
            ))

        for (chain, masked_cdrs) in self.config.cdr_mask.items():
            transforms.append(dt.normalize_coordinate(key=chain + "_int_coordinate"))
            if len(masked_cdrs) > 0:
                transforms.append(dt.make_distance(src_key=chain +  "_int_coordinate", tgt_key=chain + "_tgt_distance")) # prepare the `tgt_distance`
                transforms.append(dt.prepend_and_append_2d(key=chain + "_tgt_distance", token=0.0, width=self.config.atom_level))

        transforms.append(dt.mask_antibody_design(self.config.atom_level, self.config.cdr_mask))

        for chain in self.config.cdr_mask:
            transforms.append(dt.prepend_and_append(src_key=chain + "_src_residue", pre_token=rc.BOS, app_token=rc.EOS, repeat_mode=1, tgt_key=chain + "_src_residue")) # prepend and append (bos and eos) to prepare `src_residue`
            transforms.append(dt.make_distance(src_key=chain + "_src_coordinate", tgt_key=chain + "_src_distance")) # prepare the `src_distance`
            transforms.append(dt.prepend_and_append_2d(key=chain + "_src_distance", token=0.0, width=self.config.atom_level))
            transforms.append(dt.prepend_and_append(src_key=chain + "_src_coordinate", pre_token=0.0, app_token=0.0, repeat_mode=(self.config.atom_level, 1), tgt_key=chain + "_src_coordinate")) # prepare coordinates as `src_coordinate`
            transforms.append(dt.prepend_and_append(src_key=chain + "_tgt_residue", pre_token=rc.PAD, app_token=rc.PAD, repeat_mode=1, tgt_key=chain + "_tgt_residue")) # prepend and append (pad) to prepare `tgt_residue`

        transforms.append(dt.antibody_design_feature_postprocessing) # remove the unnecessary fields

        transforms.append(dt.cast_to_64bit_ints)
        transforms.append(dt.cast_to_32bit_floats)

        for f in transforms:
            bio_complex = f(bio_complex)

        return bio_complex


class AntibodyDesignDataFactory():
    def __init__(self, config: DictConfig):
        """
        :param config: The configuration for the data factory, majorily for the featurizer
        """

        self.featurizer = AntibodyDesignFeaturizer(config)
        self.batch_collator = SableBatchCollactor(config.atom_level)


class ModelQualityAssessmentFeaturizer():
    def __init__(self, config: DictConfig):
        """
        :param config: The configuration for the featurizer
        """

        self.config = config

    def __call__(self, bio_complex: Mapping[str, Tensor], epoch_id: int, idx: int) -> Mapping[str, Tensor]:
        """
        :param bio_complex: The original data to pass the feature extracting pipeline
        :param epoch_id: Current epoch id, starting from 0
        :param idx: The index for the datapoint, it is used for identifying datapoint and randomizing along with `epoch_id` as part of the seed
        """

        transforms = []

        transforms.append(dt.model_quality_assessment_data_preprocessing)

        if self.config.get("crop"):
            transforms.append(dt.crop_to_size(
                self.config.atom_level,
                self.config.crop.crop_size,
                self.config.crop.spatial_crop_ratio,
                seed=int(hash((self.config.get("seed", 42), (epoch_id, idx + 4))) % 999983),
                extra=[ "int_lddt" ],
            ))

        transforms.append(dt.normalize_coordinate(key="int_coordinate"))

        transforms.append(dt.prepend_and_append(src_key="int_residue", pre_token=rc.BOS, app_token=rc.EOS, repeat_mode=1, tgt_key="src_residue")) # prepend and append (bos and eos) to prepare `src_residue`
        transforms.append(dt.make_distance(src_key="int_coordinate", tgt_key="src_distance")) # prepare the `src_distance`
        transforms.append(dt.prepend_and_append_2d(key="src_distance", token=0.0, width=self.config.atom_level))
        transforms.append(dt.prepend_and_append(src_key="int_coordinate", pre_token=0.0, app_token=0.0, repeat_mode=(self.config.atom_level, 1), tgt_key="src_coordinate")) # prepare coordinates as `src_coordinate`
        transforms.append(dt.prepend_and_append(src_key="int_lddt", pre_token=0.0, app_token=0.0, repeat_mode=1, tgt_key="tgt_lddt")) # prepend and append (0.0) to prepare `tgt_lddt`

        transforms.append(dt.model_quality_assessment_feature_postprocessing) # remove the unnecessary fields

        transforms.append(dt.cast_to_64bit_ints)
        transforms.append(dt.cast_to_32bit_floats)

        for f in transforms:
            bio_complex = f(bio_complex)

        return bio_complex


class ModelQualityAssessmentDataFactory():
    def __init__(self, config: DictConfig):
        """
        :param config: The configuration for the data factory, majorily for the featurizer
        """

        self.featurizer = ModelQualityAssessmentFeaturizer(config)
        self.batch_collator = SableBatchCollactor(config.atom_level)


class BindingAffinityFeaturizer():
    def __init__(self, config: DictConfig):
        """
        :param config: The configuration for the featurizer
        """

        self.config = config

    def __call__(self, bio_complex: Mapping[str, Tensor], epoch_id: int, idx: int) -> Mapping[str, Tensor]:
        """
        :param bio_complex: The original data to pass the feature extracting pipeline
        :param epoch_id: Current epoch id, starting from 0
        :param idx: The index for the datapoint, it is used for identifying datapoint and randomizing along with `epoch_id` as part of the seed
        """

        transforms = []

        transforms.append(dt.binding_affinity_data_preprocessing)

        if self.config.get("crop"):
            transforms.append(dt.crop_to_size(
                self.config.atom_level,
                self.config.crop.crop_size,
                self.config.crop.spatial_crop_ratio,
                seed=int(hash((self.config.get("seed", 42), (epoch_id, idx + 5))) % 999983),
                extra=[ "int_mutant_coordinate", "int_mutant_residue" ],
            ))

        transforms.append(dt.binding_affinity_post_renaming) # without cropping, rename int_residue/int_coordinate to int_wild_residue/int_wild_coordinate

        for t in ["wild", "mutant"]:
            transforms.append(dt.normalize_coordinate(key="int_" + t + "_coordinate"))
            transforms.append(dt.prepend_and_append(src_key="int_" + t + "_residue", pre_token=rc.BOS, app_token=rc.EOS, repeat_mode=1, tgt_key="src_" + t + "_residue")) # prepend and append (bos and eos) to prepare `src_residue`
            transforms.append(dt.make_distance(src_key="int_" + t + "_coordinate", tgt_key="src_" + t + "_distance")) # prepare the `src_distance`
            transforms.append(dt.prepend_and_append_2d(key="src_" + t + "_distance", token=0.0, width=self.config.atom_level))
            transforms.append(dt.prepend_and_append(src_key="int_" + t + "_coordinate", pre_token=0.0, app_token=0.0, repeat_mode=(self.config.atom_level, 1), tgt_key="src_" + t + "_coordinate")) # prepare coordinates as `src_coordinate`

        transforms.append(dt.binding_affinity_feature_postprocessing) # remove the unnecessary fields

        transforms.append(dt.cast_to_64bit_ints)
        transforms.append(dt.cast_to_32bit_floats)

        for f in transforms:
            bio_complex = f(bio_complex)

        return bio_complex


class BindingAffinityDataFactory():
    def __init__(self, config: DictConfig):
        """
        :param config: The configuration for the data factory, majorily for the featurizer
        """

        self.featurizer = BindingAffinityFeaturizer(config)
        self.batch_collator = SableBatchCollactor(config.atom_level)


class EnzymeCatalyzedReactionClassificationFeaturizer():
    def __init__(self, config: DictConfig):
        """
        :param config: The configuration for the featurizer
        """

        self.config = config

    def __call__(self, bio_complex: Mapping[str, Tensor], epoch_id: int, idx: int) -> Mapping[str, Tensor]:
        """
        :param bio_complex: The original data to pass the feature extracting pipeline
        :param epoch_id: Current epoch id, starting from 0
        :param idx: The index for the datapoint, it is used for identifying datapoint and randomizing along with `epoch_id` as part of the seed
        """

        transforms = []

        transforms.append(dt.enzyme_catalyzed_reaction_classification_data_preprocessing)

        if self.config.get("crop"):
            transforms.append(dt.crop_to_size(
                self.config.atom_level,
                self.config.crop.crop_size,
                self.config.crop.spatial_crop_ratio,
                seed=int(hash((self.config.get("seed", 42), (epoch_id, idx + 6))) % 999983),
            ))

        transforms.append(dt.normalize_coordinate(key="int_coordinate"))
        transforms.append(dt.prepend_and_append(src_key="int_residue", pre_token=rc.BOS, app_token=rc.EOS, repeat_mode=1, tgt_key="src_residue")) # prepend and append (bos and eos) to prepare `src_residue`
        transforms.append(dt.make_distance(src_key="int_coordinate", tgt_key="src_distance")) # prepare the `src_distance`
        transforms.append(dt.prepend_and_append_2d(key="src_distance", token=0.0, width=self.config.atom_level))
        transforms.append(dt.prepend_and_append(src_key="int_coordinate", pre_token=0.0, app_token=0.0, repeat_mode=(self.config.atom_level, 1), tgt_key="src_coordinate")) # prepare coordinates as `src_coordinate`

        transforms.append(dt.enzyme_catalyzed_reaction_classification_feature_postprocessing) # remove the unnecessary fields

        transforms.append(dt.cast_to_64bit_ints)
        transforms.append(dt.cast_to_32bit_floats)

        for f in transforms:
            bio_complex = f(bio_complex)

        return bio_complex


class EnzymeCatalyzedReactionClassificationDataFactory():
    def __init__(self, config: DictConfig):
        """
        :param config: The configuration for the data factory, majorily for the featurizer
        """

        self.featurizer = EnzymeCatalyzedReactionClassificationFeaturizer(config)
        self.batch_collator = SableBatchCollactor(config.atom_level)


class FoldClassificationFeaturizer():
    def __init__(self, config: DictConfig):
        """
        :param config: The configuration for the featurizer
        """

        self.config = config

    def __call__(self, bio_complex: Mapping[str, Tensor], epoch_id: int, idx: int) -> Mapping[str, Tensor]:
        """
        :param bio_complex: The original data to pass the feature extracting pipeline
        :param epoch_id: Current epoch id, starting from 0
        :param idx: The index for the datapoint, it is used for identifying datapoint and randomizing along with `epoch_id` as part of the seed
        """

        transforms = []

        transforms.append(dt.fold_classification_data_preprocessing)

        if self.config.get("crop"):
            transforms.append(dt.crop_to_size(
                self.config.atom_level,
                self.config.crop.crop_size,
                self.config.crop.spatial_crop_ratio,
                seed=int(hash((self.config.get("seed", 42), (epoch_id, idx + 7))) % 999983),
            ))

        transforms.append(dt.normalize_coordinate(key="int_coordinate"))
        transforms.append(dt.prepend_and_append(src_key="int_residue", pre_token=rc.BOS, app_token=rc.EOS, repeat_mode=1, tgt_key="src_residue")) # prepend and append (bos and eos) to prepare `src_residue`
        transforms.append(dt.make_distance(src_key="int_coordinate", tgt_key="src_distance")) # prepare the `src_distance`
        transforms.append(dt.prepend_and_append_2d(key="src_distance", token=0.0, width=self.config.atom_level))
        transforms.append(dt.prepend_and_append(src_key="int_coordinate", pre_token=0.0, app_token=0.0, repeat_mode=(self.config.atom_level, 1), tgt_key="src_coordinate")) # prepare coordinates as `src_coordinate`

        transforms.append(dt.fold_classification_feature_postprocessing) # remove the unnecessary fields

        transforms.append(dt.cast_to_64bit_ints)
        transforms.append(dt.cast_to_32bit_floats)

        for f in transforms:
            bio_complex = f(bio_complex)

        return bio_complex


class FoldClassificationDataFactory():
    def __init__(self, config: DictConfig):
        """
        :param config: The configuration for the data factory, majorily for the featurizer
        """

        self.featurizer = FoldClassificationFeaturizer(config)
        self.batch_collator = SableBatchCollactor(config.atom_level)

