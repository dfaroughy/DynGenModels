import torch
import numpy as np
from torch.utils.data import Dataset
from DynGenModels.datamodules.fermi.dataprocess import FormatData
from DynGenModels.datamodules.fermi.dataprocess import PreProcessData

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
        self.summary_statistics = None
        
        ''' datasets:
            target data (t=1) :  fermi galaxy data
            source data (t=0) :  std gaussian
            '''

        self.target = self.fermi_data()
        self.source = torch.rand_like(self.target, dtype=self.target.dtype)

    def __getitem__(self, idx):
        output = {}
        output['target'] = self.fermi_preprocessing(self.target[idx]) if self.preprocess_methods is not None else self.target[idx]
        output['source'] = self.source[idx]
        return output

    def __len__(self):
        return self.target.size(0)
    
    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def fermi_data(self):
        dataset = torch.tensor(np.load(self.path+'/'+self.dataset), dtype=torch.float32)
        dataset = self.fermi_formatting(dataset)
        self.summary_statistics = self.summary_stats(dataset)
        print("INFO: loading and preprocessing data...")
        print('\t- dataset: {} \n \t- shape: {}'.format(self.path, dataset.shape))
        return dataset

    def fermi_formatting(self, sample):
        sample = FormatData(sample, cuts=self.cuts)
        sample.format()
        return sample.data
    
    def fermi_preprocessing(self, sample):
        sample = PreProcessData(data=sample, 
                                stats=self.summary_statistics,
                                methods=self.preprocess_methods
                                )
        sample.preprocess()
        return sample.galactic_features
    
    def summary_stats(self, data):
        mean = torch.mean(data, dim=0)
        std = torch.std(data, dim=0)
        min, _ = torch.min(data, dim=0)
        max, _ = torch.max(data, dim=0)
        return (mean, std, min, max)
