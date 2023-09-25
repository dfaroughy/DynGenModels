import torch
import numpy as np
from torch.utils.data import Dataset

from DynGenModels.datamodules.fermi.dataprocess import PreProcessFermiData


class FermiDataset(Dataset):

    def __init__(self, 
                 dir_path: str=None, 
                 dataset: str=None,
                 cuts: dict=None,
                 preprocess : list=None
                 ):
        
        self.path = dir_path
        self.dataset = dataset
        self.cuts = cuts
        self.preprocess_methods = preprocess 
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
        data_raw = torch.tensor(np.load(self.path+'/'+self.dataset), dtype=torch.float32)
        data = PreProcessFermiData(data_raw, cuts=self.cuts, methods=self.preprocess_methods)
        data.preprocess()
        self.summary_stats = data.summary_stats
        print("INFO: loading and preprocessing data...")
        print('\t- dataset: {} \n \t- shape: {}'.format(self.path, data.galactic_features.shape))
        return data.galactic_features
