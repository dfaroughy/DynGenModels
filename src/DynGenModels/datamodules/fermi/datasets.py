import torch
import numpy as np
from torch.utils.data import Dataset
from dataclasses import dataclass

from DynGenModels.datamodules.fermi.dataprocess import PreProcessFermiData

class FermiDataset(Dataset):

    def __init__(self, configs: dataclass):
        
        self.dataset = configs.dataset
        self.cuts = configs.cuts
        self.preprocess_methods = configs.preprocess 
        self.summary_stats = None
        
        ''' data attributes:
            - target: fermi galaxy data (theta, phi, E)
            - target_preprocessed:  fermi galaxy data with cuts and preprocessing
            - source: std gaussian noise
        '''
        self.get_target_data()
        self.get_source_data()

    def __getitem__(self, idx):
        output = {}
        output['target'] = self.target_preprocess[idx]
        output['source'] = self.source[idx]
        return output

    def __len__(self):
        return self.target.size(0)
    
    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def get_target_data(self):
        target = torch.tensor(np.load(self.dataset), dtype=torch.float32)
        target = PreProcessFermiData(target, cuts=self.cuts, methods=self.preprocess_methods)
        target.apply_cuts()
        self.target = target.features.clone()
        target.preprocess()
        self.summary_stats = target.summary_stats
        self.target_preprocess = target.features.clone()

    def get_source_data(self):
        self.source = torch.randn_like(self.target, dtype=torch.float32)
