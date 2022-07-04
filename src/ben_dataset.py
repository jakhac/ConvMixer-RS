import pandas as pd
import numpy as np
import lmdb
from pathlib import Path
from bigearthnet_patch_interface.s2_interface import BigEarthNet_S2_Patch

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch import Tensor, cat
from torch.nn.functional import interpolate

from torchvision import transforms

from sklearn.preprocessing import MultiLabelBinarizer


LABELS = [
    'Urban fabric', 'Industrial or commercial units', 'Arable land',
    'Permanent crops', 'Pastures', 'Complex cultivation patterns',
    'Land principally occupied by agriculture, with significant areas of natural vegetation',
    'Agro-forestry areas', 'Broad-leaved forest', 'Coniferous forest', 'Mixed forest',
    'Natural grassland and sparsely vegetated areas', 'Moors, heathland and sclerophyllous vegetation',
    'Transitional woodland, shrub', 'Beaches, dunes, sands', 'Inland wetlands', 'Coastal wetlands',
    'Marine waters', 'Inland waters'
]


class BenDataset(Dataset):

    def __init__(self, csv_file, lmdb_path, transforms=None, size=None, img_size=120):
        """
        Load csv and connect to lmdb file.

        Args:
            csv_file (string): path to csv file containing patch names
            lmdb_file (string): path to lmdb file containing { patch_name : data } entries
            size (int, optional): Set size to limit the dataset from csv. Defaults to 0.
        """

        assert(Path(csv_file).exists())
        assert(Path(lmdb_path).exists())

        self.img_size = img_size
        self.size = size
        self.patch_frame = pd.read_csv(Path(csv_file))
        self.lmdb_path = Path(lmdb_path)
        self.env = lmdb.open(str(self.lmdb_path), readonly=True, readahead=True, lock=False)
        
        self.transforms = transforms

        self.mlb = MultiLabelBinarizer()
        self.mlb.fit([LABELS])


    def __len__(self):
        if self.size is not None and self.size > 0 and self.size < len(self.patch_frame):
            return self.size
        
        return len(self.patch_frame)


    def __getitem__(self, idx):
        patch_name = self.patch_frame.iloc[idx, 0]

        # Retrieve data from lmdb
        with self.env.begin() as txn:
            byteflow = txn.get(patch_name.encode("utf-8"))
            s2_patch = BigEarthNet_S2_Patch.loads(byteflow)
            
        bands_10m = s2_patch.get_stacked_10m_bands()
        bands_20m = s2_patch.get_stacked_20m_bands()
        
        # Convert to tensors and interpolate lower resolution
        bands_10m_torch = Tensor(np.float32(bands_10m)).unsqueeze(dim=0)
        bands_20m_torch = Tensor(np.float32(bands_20m)).unsqueeze(dim=0)

        bands_20m_torch = interpolate(bands_20m_torch, bands_10m.shape[-2:], mode="bicubic")
        bands_stacked_torch = cat((bands_10m_torch, bands_20m_torch), 1)

        # Interpolate again, required for some models (SwinTransformer)
        bands_stacked_torch = interpolate(bands_stacked_torch, size=(self.img_size, self.img_size), mode='bicubic')

        # Normalization according to values found in:
        # https://git.tu-berlin.de/rsim/starter-kit-bigearthnet/-/blob/main/pytorch_datasets.py
        norm_transform = transforms.Normalize(mean=(
                429.9430203, 614.21682446, 590.23569706, 2218.94553375, 950.68368468, 1792.46290469, 2075.46795189,
                2266.46036911, 1594.42694882, 1009.32729131
            ), std=(
                572.41639287, 582.87945694, 675.88746967, 1365.45589904, 729.89827633, 1096.01480586, 1273.45393088,
                1356.13789355, 1079.19066363, 818.86747235
            )
        )
        bands_stacked_torch = norm_transform(bands_stacked_torch)


        str_labels = dict(s2_patch.__stored_args__.items())['new_labels']
        one_hot_labels = self.mlb.transform([str_labels])[0]

        if self.transforms:
            bands_stacked_torch[0] = self.transforms(bands_stacked_torch[0])

        return bands_stacked_torch[0], torch.from_numpy(one_hot_labels).type(torch.FloatTensor)


    def get_mlb(self):
        return self.mlb

    
    def get_str_labels(self, y):

        y = np.asarray(y)
        label_str = self.get_mlb().inverse_transform(np.asarray([y]))

        return label_str


def get_transformation_chain(version):

    # horizontal/vertical flips and random rotation
    if version == 1:
        return transforms.Compose([
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(nn.ModuleList([
                transforms.RandomRotation(90)
            ]), p=0.25)
        ])

    # version_1 + random crop with reflecting-padding
    elif version == 2:
        return transforms.Compose([
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(nn.ModuleList([
                #TODO transforms.RandomCrop(size=120, pad_if_needed=True, padding_mode='reflect'),
                transforms.RandomRotation(180) 
            ]), p=0.25)
        ])

    # version_2 + higher prob and replace RandomCrop with RandomResizedCrop
    elif version == 3:
        return transforms.Compose([
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(nn.ModuleList([
                transforms.RandomResizedCrop(size=120),
                transforms.RandomRotation(180)
            ]), p=0.4)
        ])

    # version_2 but replace reflect with constant 0
    elif version == 4:
        return transforms.Compose([
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(nn.ModuleList([
                transforms.RandomCrop(size=120, pad_if_needed=True),
                transforms.RandomRotation(180)
            ]), p=0.25)
        ])

    return None