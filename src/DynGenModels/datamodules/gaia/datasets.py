import torch
import numpy as np
from torch.utils.data import Dataset
from dataclasses import dataclass

from DynGenModels.datamodules.gaia.dataprocess import PreProcessGaiaData

class GaiaDataset(Dataset):

    def __init__(self, configs: dataclass):
        
        self.dataset = configs.dataset
        self.cuts = configs.cuts
        self.preprocess_methods = configs.preprocess 
        self.summary_stats = None
        
        ''' datasets:
            target data (t=1) :  fermi galaxy data
            source data (t=0) :  std gaussian
        '''

        self.target_preprocess = self.get_target_data()
        self.source_preprocess = self.get_source_data()

    def __getitem__(self, idx):
        output = {}
        output['target'] = self.target_preprocess[idx]
        output['source'] = self.source_preprocess[idx]
        return output

    def __len__(self):
        return self.target_preprocess.size(0)
    
    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def get_target_data(self):
        target = torch.tensor(np.load(self.dataset), dtype=torch.float32)        
        target = PreProcessFermiData(target, cuts=self.cuts, methods=self.preprocess_methods)
        target.apply_cuts()
        self.target = target.features
        target.preprocess()
        self.summary_stats = target.summary_stats
        print("INFO: loading and preprocessing data...")
        print('\t- target dataset: {} \n \t- target shape: {}'.format(self.dataset, target.features.shape))
        return target.features

    def get_covariance_data(self):
        covariance = torch.tensor(np.load(self.dataset), dtype=torch.float32)




    def get_source_data(self):
        self.source = torch.randn_like(self.target, dtype=torch.float32)
        print('\t- source dataset: std gaussian \n \t- source shape: {}'.format( self.source.shape))
        return self.source