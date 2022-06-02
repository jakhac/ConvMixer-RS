import pandas as pd
import numpy as np
import lmdb
from pathlib import Path
from bigearthnet_patch_interface.s2_interface import BigEarthNet_S2_Patch

import torch
from torch.utils.data import Dataset
from torch import Tensor, cat, stack
from torch.nn.functional import interpolate

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

    def __init__(self, csv_file, lmdb_path, size=None):
        """
        Load csv and connect to lmdb file.

        Args:
            csv_file (string): path to csv file containing patch names
            lmdb_file (string): path to lmdb file containing { patch_name : data } entries
            size (int, optional): Set size to limit the dataset from csv. Defaults to 0.
        """

        assert(Path(csv_file).exists())
        assert(Path(lmdb_path).exists())
        
        self.size = size
        self.patch_frame = pd.read_csv(Path(csv_file))
        self.lmdb_path = Path(lmdb_path)
        self.env = lmdb.open(str(self.lmdb_path), readonly=True, readahead=True, lock=False)
        
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
        
        str_labels = dict(s2_patch.__stored_args__.items())['new_labels']
        one_hot_labels = self.mlb.transform([str_labels])[0]
        
        return bands_stacked_torch[0], torch.from_numpy(one_hot_labels).type(torch.FloatTensor)
