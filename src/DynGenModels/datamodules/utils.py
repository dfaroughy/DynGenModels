import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from torch.utils.data import Dataset
from dataclasses import dataclass
from collections import namedtuple

class Data:
    def __init__(self, config):
        
        config: dataclass = config

        source: torch.Tensor = None
        target: torch.Tensor = None
        context: torch.Tensor = None
        mask: torch.Tensor = None

    def get_target(self): pass
    def get_source(self): pass
    def get_context(self): pass
    def get_mask(self): pass

class ModelDataSet(Dataset):
    def __init__(self, data: Data):

        self.dataset = data
        self.databatch = namedtuple('databatch', ['source', 'target', 'context', 'mask'])

    def __getitem__(self, idx):
        return self.databatch(source=self.dataset.source[idx],
                              target=self.dataset.target[idx],
                              context=self.dataset.context[idx],
                              mask=self.dataset.mask[idx])
    def __len__(self):
        return len(self.dataset.target)
    
    def __iter__(self):
        for idx in range(len(self)):
            yield self[idx]

class ModelDataloader:

    def __init__(self, datasets: Dataset, batch_size: int, data_split_fraction: tuple): 

        self.datasets = datasets        
        self.data_split = data_split_fraction
        self.batch_size = batch_size
        self.dataloader()

    def train_val_test_split(self, shuffle=False):
        assert np.abs(1.0 - sum(self.data_split)) < 1e-3, "Split fractions do not sum to 1!"
        total_size = len(self.datasets)
        train_size = int(total_size * self.data_split[0])
        valid_size = int(total_size * self.data_split[1])

        #...define splitting indices

        idx = torch.randperm(total_size) if shuffle else torch.arange(total_size)
        idx_train = idx[:train_size].tolist()
        idx_valid = idx[train_size : train_size + valid_size].tolist()
        idx_test = idx[train_size + valid_size :].tolist()
        
        #...Create Subset for each split

        train_set = Subset(self.datasets, idx_train)
        valid_set = Subset(self.datasets, idx_valid) if valid_size > 0 else None
        test_set = Subset(self.datasets, idx_test) if self.data_split[2] > 0 else None

        return train_set, valid_set, test_set


    def dataloader(self):

        print("INFO: building dataloaders...")
        print("INFO: train/val/test split ratios: {}/{}/{}".format(self.data_split[0], self.data_split[1], self.data_split[2]))
        
        train, valid, test = self.train_val_test_split(shuffle=True)
        self.train = DataLoader(dataset=train, batch_size=self.batch_size, shuffle=True)
        self.valid = DataLoader(dataset=valid,  batch_size=self.batch_size, shuffle=False) if valid is not None else None
        self.test = DataLoader(dataset=test,  batch_size=self.batch_size, shuffle=True) if test is not None else None

        print('INFO: train size: {}, validation size: {}, testing sizes: {}'.format(len(self.train.dataset),  # type: ignore
                                                                                    len(self.valid.dataset if valid is not None else []),  # type: ignore
                                                                                    len(self.test.dataset if test is not None else []))) # type: ignore
