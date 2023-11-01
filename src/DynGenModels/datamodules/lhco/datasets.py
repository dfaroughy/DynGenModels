import torch
import h5py
from torch.utils.data import Dataset
from dataclasses import dataclass

from DynGenModels.datamodules.lhco.dataprocess import PreProcessLHCOlympicsData

class LHCOlympicsDataset(Dataset):

    def __init__(self, configs: dataclass):

        self.dataset = configs.dataset
        self.preprocess_methods = configs.preprocess 
        self.num_dijets = configs.num_dijets
        self.cuts_sideband_low = configs.cuts_sideband_low 
        self.cuts_sideband_high = configs.cuts_sideband_high
        self.cuts_signal_region = {'mjj': [configs.cuts_sideband_low['mjj'][1], configs.cuts_sideband_high['mjj'][0]]}
        self.summary_stats = None
    
        ''' data attributes:
            - target: jetnet data
            - target_preprocessed:  jetnet data with preprocessing
            - source: std gaussian noise
        '''

        self.get_target_data()
        self.get_source_data()
        self.get_background_data()

    def __getitem__(self, idx):
        output = {}
        output['target'] = self.target_preprocess[idx]
        output['source'] = self.source_preprocess[idx]
        output['mask'] = torch.ones_like(self.target[idx][..., 0])
        output['context'] = torch.empty_like(self.target[idx][..., 0])
        return output


    def __len__(self):
        return self.target.size(0)
    
    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def get_target_data(self):
        f = h5py.File(self.dataset, 'r') 
        dijets = torch.Tensor(f['dijet features'])
        sb2 = PreProcessLHCOlympicsData(dijets, num_dijets=self.num_dijets, cuts=self.cuts_sideband_high, methods=self.preprocess_methods)
        sb2.apply_cuts()
        self.target = sb2.features.clone()
        self.target_context = sb2.log_mjj.clone()
        sb2.preprocess()
        self.target_summary_stats = sb2.summary_stats
        self.target_preprocess = sb2.features.clone()
        f.close()

    def get_source_data(self):
        f = h5py.File(self.dataset, 'r') 
        dijets = torch.Tensor(f['dijet features'])
        sb1 = PreProcessLHCOlympicsData(dijets, num_dijets=self.num_dijets, cuts=self.cuts_sideband_low, methods=self.preprocess_methods)
        sb1.apply_cuts()
        self.source = sb1.features.clone()
        self.source_context = sb1.log_mjj.clone()
        sb1.preprocess()
        self.source_summary_stats = sb1.summary_stats
        self.source_preprocess = sb1.features.clone()
        f.close()

    def get_background_data(self):
        f = h5py.File(self.dataset, 'r') 
        dijets = torch.Tensor(f['dijet features'])
        sr = PreProcessLHCOlympicsData(dijets, num_dijets=self.num_dijets, cuts=self.cuts_signal_region, methods=self.preprocess_methods)
        sr.apply_cuts(background=True)
        self.background = sr.features.clone()
        self.background_context = sr.log_mjj.clone()
        sr.preprocess()
        self.background_summary_stats = sr.summary_stats
        self.background_preprocess = sr.features.clone()
        f.close()
