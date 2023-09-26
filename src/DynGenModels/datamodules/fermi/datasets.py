import torch
import numpy as np
from torch.utils.data import Dataset
from dataclasses import dataclass

from DynGenModels.datamodules.fermi.dataprocess import PreProcessFermiData


class FermiDataset(Dataset):

    def __init__(self, config: dataclass,
                #  dir_path: str=None, 
                #  dataset: str=None,
                #  cuts: dict=None,
                #  preprocess : list=None
                 ):
        
        self.dataset = config.dataset
        self.cuts = config.cuts
        self.preprocess_methods = config.preprocess 
        self.summary_stats = None
        
        ''' datasets:
            target data (t=1) :  fermi galaxy data
            source data (t=0) :  std gaussian
        '''

        self.target = self.get_fermi_data()
        self.source = torch.randn_like(self.target, dtype=self.target.dtype)

    def __getitem__(self, idx):
        output = {}
        output['target'] = self.target[idx]
        output['source'] = self.source[idx]
        return output

    def __len__(self):
        return self.target.size(0)
    
    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def get_fermi_data(self):
        data_raw = torch.tensor(np.load(self.dataset), dtype=torch.float32)
        data = PreProcessFermiData(data_raw, cuts=self.cuts, methods=self.preprocess_methods)
        data.preprocess()
        self.summary_stats = data.summary_stats
        print("INFO: loading and preprocessing data...")
        print('\t- dataset: {} \n \t- shape: {}'.format(self.dataset, data.galactic_features.shape))
        return data.galactic_features
