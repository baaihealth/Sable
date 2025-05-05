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

import base64
from functools import partial
import random
from typing import Any, Callable, Mapping, Sequence, Tuple, Union

from omegaconf import DictConfig
from torch import Tensor
import torch

from opencomplex.data.data_transforms import curry1
from sable.np import residue_constant as rc


def sable_data_preprocessing(bio_complex: Mapping[str, Tensor]) -> Mapping[str, Tensor]:
    """
    Preprocessing procedure does:
        1. Map the "residues" each from characters to numerical IDs
        2. Flatten the "coordinates" field

    :param bio_complex: The data to be passed through and processed by the feature extracting pipeline
    """

    bio_complex["int_residue"] = torch.IntTensor(list(map(lambda x: rc.residue_id_lookup(x), bio_complex["residues"]))) # map residues to IDs
    bio_complex["int_coordinate"] = torch.FloatTensor(bio_complex["coordinates"][0]) # "coordinates" is a list of tensors with length 1, pull it out
    return bio_complex


def random_crop_to_size_multiple_chains(chainid: Sequence[Any], fixed_offset: Sequence[int], crop_size: int, randint: Callable[[int, int], int]) -> Tensor:
    """
    Crop multiple chains protein to one with length `cro_size`, it assumes that residues from the same chain are successive in the sequence
    It tries to make sure all chains have residues being kept, and consider the protein as single chain when number of chains is larger than the crop size
    The kept indices are returned

    :param chainid: The chain information for each residue
    :param fixed_offset: The offsets of residue to be deterministically selected
    :param crop_size: The number of protein residues left after cropping
    :param randint: The simplified random partial function with only parameters of boundaries
    """

    chains = [] # store the information of each chain, including length of this chain and fixed offsets inside
    p = 0
    for i in range(len(chainid)):
        if i == 0 or chainid[i] != chainid[i - 1]: # new chain
            chains.append([1, []])
        else:
            chains[-1][0] += 1
        if p < len(fixed_offset) and fixed_offset[p] == i: # add internal offset
            chains[-1][1].append(fixed_offset[p])
            p += 1
    if sum(len(x[1]) == 0 for x in chains) + len(fixed_offset) > crop_size: # too many chains
        chains = [[len(chainid), fixed_offset]]

    while True:
        remain_length = crop_size # count the remain length
        max_segment_chain_start = -1
        max_segment_offset = -1
        max_segment_length = 0
        p = 0
        for i, chain in enumerate(chains):
            if len(chain[1]) == 0:
                remain_length -= 1
            else:
                l = chain[1][-1] - chain[1][0] + 1
                if l > max_segment_length:
                    max_segment_chain_start = p
                    max_segment_offset = i
                    max_segment_length = l
                remain_length -= l
            p += chain[0]
        if remain_length >= 0: # the remain length is enough
            break
        fixed = chains[max_segment_offset][1] # not enough, try to split the max segment into two segments
        p = (fixed[0] + fixed[-1]) // 2 # take the middle and splitting position (belongs to the segment in the front itself)
        l = p - max_segment_chain_start + 1
        chains.insert(max_segment_offset + 1, [chains[max_segment_offset][0] - l, list(filter(lambda x: x > p, fixed))])
        chains[max_segment_offset] = [l, list(filter(lambda x: x <= p, fixed))]

    free_residues = [] # pool to select extra residue from
    extra_residues = {}
    for i, chain in enumerate(chains):
        if len(chain[1]) == 0:
            extra_residues[i] = 1
            free_residues += [i for _ in range(chain[0] - 1)]
        else:
            extra_residues[i] = 0
            free_residues += [i for _ in range(chain[0] - (chain[1][-1] - chain[1][0] + 1))]
    for i in range(remain_length): # randomly sample `remain_length` residues from `free_residues` in shuffle progress
        k = int(randint(i, len(free_residues)))
        if k != i:
            free_residues[i], free_residues[k] = free_residues[k], free_residues[i]
        extra_residues[free_residues[i]] += 1

    offset = 0
    indices = [] # decide the cropping offset and obtain indices for residue and coordinate
    for i, chain in enumerate(chains):
        if len(chain[1]) == 0: # no fixed offset, random pick freely
            select_length = extra_residues[i]
            select_offset = offset + int(randint(0, chain[0] - select_length + 1))
        else: # the selected part should cover all fixed offsets
            select_length = extra_residues[i] + chain[1][-1] - chain[1][0] + 1
            select_offset = int(randint(max(offset, chain[1][-1] - select_length + 1), min(offset + chain[0] - select_length, chain[1][0]) + 1))
        indices.append(torch.arange(select_offset, select_offset + select_length))
        offset += chain[0]

    return torch.cat(indices)


def spatial_crop_to_size_multiple_chains(chainid: Sequence[Any], fixed_offset: Sequence[int], c_alpha: Tensor, crop_size: int, randint: Callable[[int, int], int]) -> Sequence[int]:
    """
    Spatial cropping to `crop_size`, or keep as is if shorter than that.
    Implements Alphafold Multimer Algorithm 2, but make all homo chains have same cropping area.

    :param chainid: The chain information for each residue
    :param fixed_offset: The offsets of residue to be deterministically selected
    :param c_alpha: The coordinate of $C_\alpha$ for each residue
    :param crop_size: The number of protein residues left after cropping
    :param randint: The simplified random partial function with only parameters of boundaries
    """

    pdist = torch.pairwise_distance(c_alpha[..., None, :], c_alpha[..., None, :, :])
    chainid_dict = list(set(chainid))
#    if len(chainid_dict) > crop_size:
#        asym_id = torch.zeros(len(chainid)) # for too many chains, consider them as a single chain
#    else:
#        asym_id = torch.IntTensor(list(map(lambda x: chainid_dict.index(x), chainid))) # encode chain IDs that each into
    asym_id = torch.IntTensor(list(map(lambda x: chainid_dict.index(x), chainid))) # encode chain IDs that each into
    inter_chain_mask = ~(asym_id[..., :, None] == asym_id[..., None, :]) # mask that whether two residues are from different chains
    interface_residue = torch.logical_and(pdist <= 6.0, inter_chain_mask) # find interface residue pair
#    interface_residue = torch.any(interface_residue, dim=-1)
    interface_residue = torch.flatten(torch.nonzero(interface_residue)) # get interface residues

    if interface_residue.shape[-1] == 0:
        return None

    selected_interface_residue = interface_residue[int(randint(0, interface_residue.shape[-1]))] # randomly pick one interface residue
    center_pos = c_alpha[selected_interface_residue] # use the randomly picked interface residue as center
    center_dist = torch.pairwise_distance(c_alpha, center_pos) # obtain distances to the center

    all_candidates = torch.argsort(center_dist)[:crop_size].tolist() # pick the closest ones
    forbidden = set(fixed_offset)
    ret = fixed_offset # the fixed offsets are sure to be selected and construct the forbidden set
    for x in all_candidates:
        if not(x in forbidden):
            ret.append(x)
            if len(ret) >= crop_size:
                break
    return sorted(ret)


@curry1
def crop_to_size(bio_complex: Mapping[str, Any], atom_level: int, crop_size: int, spatial_crop_ratio: float, seed: int=None, prefix: str='', extra: Sequence[str]=[]) -> Mapping[str, Any]:
    """
    Crop multi-chain protein that trying to make sure each chain has segment been kept
    For proteins with too many chains, treat them as single-chain ones

    :param bio_complex: The data to be passed through and processed by the feature extracting pipeline
    :param atom_level: Number of atom coordinates provided in a protein residue
    :param crop_size: The number of protein residues left after cropping
    :param spatial_crop_ratio: The probability to apply spatial cropping
    :param seed: The random seed for reproducing
    """

    chainid = bio_complex[prefix + "chainid"]

    if len(chainid) <= crop_size: # short enough that there is no need to crop
        return bio_complex

    fixed_offset = bio_complex["int_mutation_site"].tolist() if "int_mutation_site" in bio_complex else []
    g = torch.Generator()
    g.manual_seed(seed)
    device = bio_complex[prefix + "int_residue"].device
    randint = partial(torch.randint, size=(), device=device, generator=g)
    if float(torch.rand((), generator=g, device=device)) > spatial_crop_ratio or len(chainid) > 8000: # equivalent to check 8000 length first and set spatial_crop_ratio < 0
        if float(torch.rand((), generator=g, device=device)) > 0.5 or len(set(chainid)) > crop_size: # equivalent to catch exception and then set as single chain
            chainid = ["A" for _ in range(len(chainid))]
        indices = random_crop_to_size_multiple_chains(chainid, fixed_offset, crop_size, randint)
    else:
        indices = spatial_crop_to_size_multiple_chains(chainid, fixed_offset, bio_complex[prefix + "int_coordinate"][1 : : atom_level], crop_size, randint)
        if indices is None:
            indices = random_crop_to_size_multiple_chains(chainid, fixed_offset, crop_size, randint)

    bio_complex[prefix + "int_residue"] = bio_complex[prefix + "int_residue"][indices] # crop the default two fields
    bio_complex[prefix + "int_coordinate"] = bio_complex[prefix + "int_coordinate"].view(-1, atom_level, 3)[indices].view(-1, 3)
    for k in extra: # crop extra fields
        if '.' in k:
            i = k.index('.')
            data = bio_complex[k[ : i]]
            name = k[i + 1 : ]
        else:
            data = bio_complex
            name = k
        data[name] = data[name][indices] if not("_coordinate" in name) else data[name].view(-1, atom_level, 3)[indices].view(-1, 3)

    return bio_complex


@curry1
def normalize_coordinate(bio_complex: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    """
    Normalize the coordinates that the mean coordinate along each axis is 0

    :param bio_complex: The data to be passed through and processed by the feature extracting pipeline
    :param key: Identifying the field to apply coordinate normalization
    """

    if key in bio_complex:
        coordinate = bio_complex[key]
        bio_complex[key] = coordinate - coordinate.mean(axis=0)
    return bio_complex


@curry1
def mask_protein(bio_complex: Mapping[str, Tensor], atom_level: int, fake_p: float, kept_p: float, seed: int=None) -> Mapping[str, Tensor]:
    """
    Mask and disturb residues
    1. Mask some residues first, noises are added to all coordinates related
    2. In those masked residues
        a. Pick some to keep unmasked, with probability `kept_p`
        b. Pick some to assign a random residue, with probability `fake_p`

    :param bio_complex: The data to be passed through and processed by the feature extracting pipeline
    :param atom_level: Number of atom coordinates provided in a protein residue
    :param fake_p: The probability to randomize a residue
    :param kept_p: The probability to leave residue unmasked
    :param seed: The random seed for reproducing
    """

    residue = bio_complex["int_residue"]
    sz = len(residue)

    g = torch.Generator(device=residue.device)
    if seed is not None:
        g.manual_seed(seed)
    noise_f = lambda n, r: torch.normal(0, r, size=(n, 3), generator=g)

    p = float(torch.rand((1, ), generator=g)[0])
    if p <= 0.5:
        mask_p, fake_p = 1, 0.1
    elif p <= 0.7:
        mask_p, fake_p = 0.5, 0.1
    else:
        mask_p = 0.15

    num_mask = int(torch.clamp(mask_p * sz + torch.rand((1, ), generator=g)[0], min=1, max=sz)) # add a random number for probabilistic rounding
    mask_indices = torch.multinomial(torch.ones((sz, )), num_mask, replacement=False, generator=g) # randomly choose `num_mask` residues with their indices
    mask = torch.full((sz, ), False)
    mask[mask_indices] = True

    tgt_residue = torch.full_like(residue, rc.PAD) # non-[MASK] residues are labeled as [PAD] in the target
    tgt_residue[mask] = residue[mask]
    bio_complex["tgt_residue"] = tgt_residue

    if abs(fake_or_kept_p := fake_p + kept_p) < 1e-5:
        fake_mask, kept_mask = None, None
    else:
        fake_or_kept_mask = mask & (torch.rand((sz, ), generator=g) < fake_or_kept_p) # randomize/keep parts should be in the region of mask
        if abs(fake_p) < 1e-5:
            fake_mask, kept_mask = None, fake_or_kept_mask
        elif abs(kept_p) < 1e-5:
            fake_mask, kept_mask = fake_or_kept_mask, None
        else:
            unmasked_p = kept_p / fake_or_kept_p
            decision = torch.rand((sz, ), generator=g) < unmasked_p
            fake_mask = fake_or_kept_mask & (~decision)
            kept_mask = fake_or_kept_mask & decision

    if not(kept_mask is None):
        mask = mask ^ kept_mask

    residue[mask] = rc.MASK # in mask, ones other than randomize/keep are masked

    if atom_level > 4 and float(torch.rand(1, generator=g)) <= 0.3: # used by data with 5 atoms per residue
        coordinate_mask = torch.cat((torch.full_like(mask, False).repeat_interleave(4).view(-1, 4), mask.repeat_interleave(atom_level - 4).view(-1, atom_level - 4)), dim=1).flatten()
    else:
        coordinate_mask = mask.repeat_interleave(atom_level)
    coordinate = bio_complex["int_coordinate"]
    noise_p = float(torch.rand(1, generator=g))
    if noise_p <= 0.2: # all coordinates are disturbed before unmask/randomize opration
        coordinate[coordinate_mask, : ] += noise_f(int(coordinate_mask.int().sum()), 0.1)
    elif noise_p <= 0.8:
        coordinate[coordinate_mask, : ] += noise_f(int(coordinate_mask.int().sum()), 1.0)
    bio_complex["src_coordinate"] = coordinate

    if not(fake_mask is None):
        fake_count = int(fake_mask.int().sum())
        if fake_count > 0:
            residue[fake_mask] = torch.multinomial(rc.restype_weights, fake_count, replacement=True, generator=g).to(residue.dtype)
    bio_complex["src_residue"] = residue

    return bio_complex


@curry1
def prepend_and_append(bio_complex: Mapping[str, Any], src_key: str, pre_token: Union[int, float], app_token: Union[int, float], repeat_mode: Union[int, Tuple[int, int]], tgt_key: str) -> Mapping[str, Any]:
    """
    Prepend and append tokens for both 1D and 2D cases
        1. for 1D data like residue sequence, calls `PrependAndAppend(item, pre_token, app_token, 1)`
        2. for 2D data like coordinates, call `PrependAndAppend(item, pre_token, app_token, (atom_level, 1))`

    :param bio_complex: The data to be passed through and processed by the feature extracting pipeline
    :param src_key: The key for field to be prepended and appended
    :param pre_token: The token value for prepending
    :param app_token: The token value for appending
    :param repeat_mode: Different operating repeat mode base on different dimensionality of data
    :param tgt_key: The key for generated field to be prepended and appended
    """

    if src_key in bio_complex:
        item = bio_complex[src_key].detach() # use a copy of item here to avoid value update
        bio_complex[tgt_key] = torch.cat([torch.full_like(item[0], pre_token).repeat(repeat_mode), item, torch.full_like(item[0], app_token).repeat(repeat_mode)], dim=0)
    return bio_complex


@curry1
def make_distance(bio_complex: Mapping[str, Any], src_key: str, tgt_key: str) -> Mapping[str, Any]:
    """
    Calculate the distance matrix

    :param bio_complex: The data to be passed through and processed by the feature extracting pipeline
    :param src_key: The key for field to be prepended and appended
    :param tgt_key: The key for generated field to be prepended and appended
    """

    if src_key in bio_complex:
        coordinate = bio_complex[src_key]
        bio_complex[tgt_key] = torch.cdist(coordinate, coordinate, p=2, compute_mode="donot_use_mm_for_euclid_dist")
    return bio_complex


@curry1
def prepend_and_append_2d(bio_complex: Mapping[str, Any], key: str, token: float, width: int) -> Mapping[str, Any]:
    """
    The 2D version of prepend and append operation, which adds a "border" of width `width`

    :param bio_complex: The data to be passed through and processed by the feature extracting pipeline
    :param key: The key for field to be prepended and appended
    :param token: The token value for prepending and appending
    :param width: The width of the border, the border is added to both end
    """

    if key in bio_complex:
        distance_matrix = bio_complex[key]
        new_distance_matrix = torch.full((len(distance_matrix) + 2 * width, len(distance_matrix) + 2 * width), token, dtype=distance_matrix.dtype)
        new_distance_matrix[width : -width, width : -width] = distance_matrix
        bio_complex[key] = new_distance_matrix
    return bio_complex


def sable_feature_postprocessing(bio_complex: Mapping[str, Tensor]) -> Mapping[str, Tensor]:
    """
    Remove the unnecessary fields to make the features compact

    :param bio_complex: The data to be passed through and processed by the feature extracting pipeline
    """

    bio_complex.pop("pdbid", 0)
    bio_complex.pop("residues", 0)
    bio_complex.pop("chainid", 0)
    bio_complex.pop("coordinates", 0)
    bio_complex.pop("int_residue", 0)
    bio_complex.pop("int_coordinate", 0)
    return bio_complex


@curry1
def feature_padding(features: Sequence[Tensor], atom_level: int=1, left_pad: bool=False, pad_token: Union[int, float]=0, pad_to_multiple: int=8) -> Mapping[str, Tensor]:
    """
    Pad feature to be a batch, which works for both 1D sequence and 2D coordinates

    :param features: The data to be passed through and processed by the feature extracting pipeline
    :param atom_level: Number of atom coordinates provided in a protein residue
    :param left_pad: The indicator for whether data is padded to the left
    :param pad_token: The token value for the padding slots
    :param pad_to_multiple: The size that padding is aligned with
    """

    size = max([x.size(0) for x in features]) // atom_level
    size = (size + pad_to_multiple - 1) // pad_to_multiple * pad_to_multiple * atom_level # apply the pad_to_multiple
    ret = features[0].new_full(sum(((len(features), size), (features[0].shape[1:])), ()), pad_token) # prepare the result tensor, works for both 1D sequence and 2D coordinates
    for i, v in enumerate(features):
        if left_pad:
            ret[i, size - len(v) : ].copy_(v)
        else:
            ret[i, : len(v)].copy_(v)
    return ret


@curry1
def feature_padding_2d(features: Sequence[Tensor], atom_level: int=1, left_pad: bool=False, pad_token: Union[int, float]=0, pad_to_multiple: int=8) -> Sequence[Tensor]:
    """
    Pad feature to be a batch, which works for distance matrix

    :param features: The data to be passed through and processed by the feature extracting pipeline
    :param atom_level: Number of atom coordinates provided in a protein residue
    :param left_pad: The indicator for whether data is padded to the left
    :param pad_token: The token value for the padding slots
    :param pad_to_multiple: The size that padding is aligned with
    """

    size = max([x.size(0) for x in features]) // atom_level
    size = (size + pad_to_multiple - 1) // pad_to_multiple * pad_to_multiple * atom_level # apply the pad_to_multiple
    ret = features[0].new_full((len(features), size, size), pad_token) # prepare the result tensor, as a square tensor
    for i, v in enumerate(features):
        if left_pad:
            ret[i, size - len(v) : , size - len(v) :].copy_(v)
        else:
            ret[i, : len(v), : len(v)].copy_(v)
    return ret


def cast_to_64bit_ints(bio_complex: Mapping[str, Any]) -> Mapping[str, Any]:
    """
    Keep all ints as int64

    :param bio_complex: The data to be passed through and processed by the feature extracting pipeline
    """

    for k, v in bio_complex.items():
        if "pdb" in k:
            continue
        if v.dtype == torch.int32:
            bio_complex[k] = v.type(torch.int64)

    return bio_complex


def cast_to_32bit_floats(bio_complex: Mapping[str, Any]) -> Mapping[str, Any]:
    """
    Keep all floats at most float32

    :param bio_complex: The data to be passed through and processed by the feature extracting pipeline
    """

    for k, v in bio_complex.items():
        if "pdb" in k:
            continue
        if v.dtype == torch.float64:
            bio_complex[k] = v.type(torch.float32)

    return bio_complex


def protein_design_data_preprocessing(bio_complex: Mapping[str, Tensor]) -> Mapping[str, Tensor]:
    """
    Preprocessing procedure does:
        1. Map the "residues" each from characters to numerical IDs
        2. Flatten the "coordinates" field

    :param bio_complex: The data to be passed through and processed by the feature extracting pipeline
    """

    bio_complex["tgt_residue"] = torch.IntTensor(list(map(lambda x: rc.residue_id_lookup(x), bio_complex["residues"]))) # map residues to IDs
    bio_complex["int_residue"] = torch.full((len(bio_complex["residues"]), ), rc.MASK) # everything is set as [MASK]
    bio_complex["int_coordinate"] = torch.FloatTensor(bio_complex["coordinates"][0]) # "coordinates" is a list of tensors with length 1, pull it out
    return bio_complex


def protein_design_feature_postprocessing(bio_complex: Mapping[str, Tensor]) -> Mapping[str, Tensor]:
    """
    Remove the unnecessary fields to make the features compact

    :param bio_complex: The data to be passed through and processed by the feature extracting pipeline
    """

    bio_complex.pop("pdbid", 0)
    bio_complex.pop("residues", 0)
    bio_complex.pop("chainid", 0)
    bio_complex.pop("coordinates", 0)
    bio_complex.pop("int_residue", 0)
    bio_complex.pop("int_coordinate", 0)
    return bio_complex


@curry1
def antibody_design_data_preprocessing(bio_complex: Mapping[str, Any], cdr_mask: DictConfig) -> Mapping[str, Any]:
    """
    For each chain, preprocessing procedure does:
        1. Map the "residues" each from characters to numerical IDs
        2. Flatten the "coordinates" field
    And for the heavy and light chains, make CDR values into numerical ones

    :param bio_complex: The data to be passed through and processed by the feature extracting pipeline
    :param cdr_mask: The CDR information for components to be processed
    """

    for chain in cdr_mask:
        if not((chain + "_residues") in bio_complex):
            continue
        bio_complex[chain + "_int_coordinate"] = torch.FloatTensor(bio_complex[chain + "_coordinates"][0]) # "coordinates" is a list of tensors with length 1, pull it out
        bio_complex[chain + "_int_residue"] = torch.IntTensor(list(map(lambda x: rc.residue_id_lookup(x), bio_complex[chain + "_residues"]))) # map residues to IDs as targets
        if chain != "Ag": # only heavy chains and light chains have CDR
            bio_complex[chain + "_pdb"]["cdr"] = torch.IntTensor(list(map(lambda x: int(x), bio_complex[chain + "_cdr"]))) # get the CDR IDs for heavy chain
    return bio_complex


@curry1
def mask_antibody_design(bio_complex: Mapping[str, Any], atom_level: int, cdr_mask: DictConfig) -> Mapping[str, Any]:
    """
    Mask CDRs in the antibody heavy/light chains, base on information from `cdr_mask`

    :param bio_complex: The data to be passed through and processed by the feature extracting pipeline
    :param atom_level: Number of atom coordinates provided in a protein residue
    :param cdr_mask: The CDR information for components to be processed
    """

    for (chain, masked_cdrs) in cdr_mask.items(): # only heavy chain and light chain have CDRs, but obtain information together
        src_coordinate = bio_complex[chain + "_int_coordinate"].detach().clone()
        src_residue = bio_complex[chain + "_int_residue"].detach().clone()

        if chain != "Ag": # mask CDRs, and antigen does not have "Ag_cdr"
            cdr = bio_complex[chain + "_pdb"]["cdr"]

            for masked_cdr in masked_cdrs:
                mask = (cdr == masked_cdr) # take one of the CDRs to mask
                if int(mask.sum()) > 0:
                    src_residue[mask] = rc.MASK # mask related CDR residues with [MASK]
                    mask_indices = torch.where(mask)
                    cdr_coordinate_indices = []
                    if int(mask_indices[0][0]) > 0: # left non-mask exists
                       cdr_coordinate_indices.append((int(mask_indices[0][0]) - 1) * atom_level + 1) # the coordinate of previous C_alpha will be used
                    if int(mask_indices[0][-1]) + 1 < len(cdr): # right non-mask exists
                       cdr_coordinate_indices.append((int(mask_indices[0][-1]) + 1) * atom_level + 1) # the coordinate of next C_alpha will be used
                    cdr_coordinate = src_coordinate[cdr_coordinate_indices].mean(axis=0) # take the mean coordinates as the masked CDR coordinate
                    src_coordinate[mask.repeat_interleave(atom_level)] = cdr_coordinate # replace the coordinates

            mask = (src_residue == rc.MASK)
            if mask.any():
                tgt_residue = torch.full_like(src_residue, rc.PAD) # set the whole target as [PAD]
                tgt_residue[mask] = bio_complex[chain + "_int_residue"][mask]
                bio_complex[chain + "_tgt_residue"] = tgt_residue

        bio_complex[chain + "_src_coordinate"] = src_coordinate
        bio_complex[chain + "_src_residue"] = src_residue
    return bio_complex


def antibody_design_feature_postprocessing(bio_complex: Mapping[str, Any]) -> Mapping[str, Any]:
    """
    Remove the unnecessary fields to make the features compact

    :param bio_complex: The data to be passed through and processed by the feature extracting pipeline
    """

    # no need to care about padding here since they are only used in loss
    for chain in ["H", "L", "Ag"]:
        bio_complex.pop(chain + "_cdr", 0)
        bio_complex.pop(chain + "_coordinates", 0)
        bio_complex.pop(chain + "_residues", 0)
        bio_complex.pop(chain + "_int_coordinate", 0)
        bio_complex.pop(chain + "_int_residue", 0)
    bio_complex.pop("pdbid", 0)
    bio_complex.pop("Ag_chainid", 0)
    return bio_complex


def model_quality_assessment_data_preprocessing(bio_complex: Mapping[str, Tensor]) -> Mapping[str, Tensor]:
    """
    Preprocessing procedure does:
        1. Map the "residues" each from characters to numerical IDs
        2. Flatten the "coordinates" field
        3. Convert the ground truth to tensor

    :param bio_complex: The data to be passed through and processed by the feature extracting pipeline
    """

    bio_complex["int_coordinate"] = torch.FloatTensor(bio_complex["coordinates"][0]) # "coordinates" is a list of tensors with length 1, pull it out
    bio_complex["tgt_gdt_ts"] = torch.tensor(bio_complex["gdt_ts"]).float() # "gdt_ts" is a float, make it a tensor
    bio_complex["int_lddt"] = torch.FloatTensor(bio_complex["lddt"][0]) # "lddt" is a list of tensors with length 1, pull it out
    bio_complex["int_residue"] = torch.IntTensor(list(map(lambda x: rc.residue_id_lookup(x), bio_complex["residues"]))) # map residues to IDs
    bio_complex["id"] = torch.tensor(int.from_bytes(base64.b64decode(bio_complex["pdbid"].split("_")[0] + "B=="), "little")) # keep the PDB ID in integer value for grouping, BASE64 is a good way of decoding
    return bio_complex


def model_quality_assessment_feature_postprocessing(bio_complex: Mapping[str, Tensor]) -> Mapping[str, Tensor]:
    """
    Remove the unnecessary fields to make the features compact

    :param bio_complex: The data to be passed through and processed by the feature extracting pipeline
    """

    bio_complex.pop("chainid", 0)
    bio_complex.pop("coordinates", 0)
    bio_complex.pop("gdt_ts", 0)
    bio_complex.pop("global_lddt", 0)
    bio_complex.pop("int_coordinate", 0)
    bio_complex.pop("int_lddt", 0)
    bio_complex.pop("int_residue", 0)
    bio_complex.pop("lddt", 0)
    bio_complex.pop("pdbid", 0)
    bio_complex.pop("residues", 0)
    return bio_complex


def binding_affinity_data_preprocessing(bio_complex: Mapping[str, Tensor]) -> Mapping[str, Tensor]:
    """
    Preprocessing procedure does:
        1. Map the "residues" each from characters to numerical IDs
        2. Flatten the "coordinates" field
        3. Convert the ground truth "binding_free_energy_change" to tensor

    :param bio_complex: The data to be passed through and processed by the feature extracting pipeline
    """

    bio_complex["int_coordinate"] = torch.FloatTensor(bio_complex["wild_coordinates"][0]) # "coordinates" is a list of tensors with length 1, pull it out
    bio_complex["int_residue"] = torch.IntTensor(list(map(lambda x: rc.residue_id_lookup(x), bio_complex["wild_residues"]))) # map residues to IDs
    bio_complex["int_mutant_coordinate"] = torch.FloatTensor(bio_complex["mutant_coordinates"][0]) # "coordinates" is a list of tensors with length 1, pull it out
    bio_complex["int_mutant_residue"] = torch.IntTensor(list(map(lambda x: rc.residue_id_lookup(x), bio_complex["mutant_residues"]))) # map residues to IDs
    bio_complex["int_mutation_site"] = torch.where(torch.ne(bio_complex["int_residue"], bio_complex["int_mutant_residue"]))[0] # all the offsets of mutation site
    bio_complex["tgt_ddg"] = torch.tensor(bio_complex["binding_free_energy_change"]).float() # "binding_free_energy_change" is a float, make it a tensor
    return bio_complex


def binding_affinity_post_renaming(bio_complex: Mapping[str, Tensor]) -> Mapping[str, Tensor]:
    bio_complex["int_wild_coordinate"] = bio_complex["int_coordinate"]
    bio_complex["int_wild_residue"] = bio_complex["int_residue"]
    return bio_complex


def binding_affinity_feature_postprocessing(bio_complex: Mapping[str, Tensor]) -> Mapping[str, Tensor]:
    """
    Remove the unnecessary fields to make the features compact

    :param bio_complex: The data to be passed through and processed by the feature extracting pipeline
    """

    bio_complex.pop("binding_free_energy_change", 0)
    bio_complex.pop("chainid", 0)
    bio_complex.pop("int_coordinate", 0)
    bio_complex.pop("int_residue", 0)
    bio_complex.pop("int_mutant_coordinate", 0)
    bio_complex.pop("int_mutant_residue", 0)
    bio_complex.pop("int_wild_coordinate", 0)
    bio_complex.pop("int_wild_residue", 0)
    bio_complex.pop("int_mutation_site", 0)
    bio_complex.pop("mutant_coordinates", 0)
    bio_complex.pop("mutant_residues", 0)
    bio_complex.pop("partners", 0)
    bio_complex.pop("pdbid", 0)
    bio_complex.pop("wild_coordinates", 0)
    bio_complex.pop("wild_residues", 0)
    return bio_complex


def enzyme_catalyzed_reaction_classification_data_preprocessing(bio_complex: Mapping[str, Tensor]) -> Mapping[str, Tensor]:
    """
    Preprocessing procedure does:
        1. Map the "residues" each from characters to numerical IDs
        2. Flatten the "coordinates" field
        3. Convert the "class" labels into tensor

    :param bio_complex: The data to be passed through and processed by the feature extracting pipeline
    """

    bio_complex["int_residue"] = torch.IntTensor(list(map(lambda x: rc.residue_id_lookup(x), bio_complex["residues"]))) # map residues to IDs
    bio_complex["int_coordinate"] = torch.FloatTensor(bio_complex["coordinates"][0]) # "coordinates" is a list of tensors with length 1, pull it out
    bio_complex["tgt_class"] = torch.tensor(int(bio_complex["class"])).int() # "class" is an integer, pull it out
    return bio_complex


def enzyme_catalyzed_reaction_classification_feature_postprocessing(bio_complex: Mapping[str, Tensor]) -> Mapping[str, Tensor]:
    """
    Remove the unnecessary fields to make the features compact

    :param bio_complex: The data to be passed through and processed by the feature extracting pipeline
    """

    bio_complex.pop("chainid", 0)
    bio_complex.pop("class", 0)
    bio_complex.pop("coordinates", 0)
    bio_complex.pop("int_coordinate", 0)
    bio_complex.pop("int_residue", 0)
    bio_complex.pop("pdbid", 0)
    bio_complex.pop("residues", 0)
    return bio_complex


def fold_classification_data_preprocessing(bio_complex: Mapping[str, Tensor]) -> Mapping[str, Tensor]:
    """
    Preprocessing procedure does:
        1. Map the "residues" each from characters to numerical IDs
        2. Flatten the "coordinates" field
        3. Convert the "class" labels into tensor

    :param bio_complex: The data to be passed through and processed by the feature extracting pipeline
    """

    bio_complex["int_residue"] = torch.IntTensor(list(map(lambda x: rc.residue_id_lookup(x), bio_complex["residues"]))) # map residues to IDs
    bio_complex["int_coordinate"] = torch.FloatTensor(bio_complex["coordinates"][0]) # "coordinates" is a list of tensors with length 1, pull it out
    bio_complex["tgt_class"] = torch.tensor(int(bio_complex["class"])).int() # "class" is an integer, pull it out
    return bio_complex


def fold_classification_feature_postprocessing(bio_complex: Mapping[str, Tensor]) -> Mapping[str, Tensor]:
    """
    Remove the unnecessary fields to make the features compact

    :param bio_complex: The data to be passed through and processed by the feature extracting pipeline
    """

    bio_complex.pop("chainid", 0)
    bio_complex.pop("class", 0)
    bio_complex.pop("coordinates", 0)
    bio_complex.pop("int_coordinate", 0)
    bio_complex.pop("int_residue", 0)
    bio_complex.pop("pdbid", 0)
    bio_complex.pop("residues", 0)
    return bio_complex

