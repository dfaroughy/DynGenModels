import torch
import h5py
from torch.utils.data import Dataset
from dataclasses import dataclass

from DynGenModels.datamodules.lhco.dataprocess import PreProcessLHCOlympicsLowLevelData, PreProcessLHCOlympicsHighLevelData


class LHCOlympicsHighLevelDataset(Dataset):

    def __init__(self, configs: dataclass, exchange_target_with_source=False):

        self.dataset = configs.dataset
        self.preprocess_methods = configs.preprocess 
        self.num_dijets = configs.num_dijets
        self.cuts_sideband_low = configs.cuts_sideband_low 
        self.cuts_sideband_high = configs.cuts_sideband_high
        self.cuts_signal_region = {'mjj': [configs.cuts_sideband_low['mjj'][1], configs.cuts_sideband_high['mjj'][0]]}
        self.cuts_sideband_region = {'mjj': [configs.cuts_sideband_low['mjj'][0], configs.cuts_sideband_high['mjj'][1]]}
        self.summary_stats = None
        self.exchange_target_with_source = exchange_target_with_source
    
        ''' data attributes:

            (label, mjj, mj1, delta_m, tau21_1, tau21_2)

            - target: SB2 data
            - source: SB1 data
        '''
        self.get_stats()
        self.get_source_data()
        self.get_target_data()
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

    def get_stats(self):
        f = h5py.File(self.dataset, 'r') 
        dijets = torch.Tensor(f['jet features'])
        sidebands = PreProcessLHCOlympicsHighLevelData(dijets, methods=self.preprocess_methods)
        sidebands.apply_cuts(cuts=self.cuts_sideband_region)
        sidebands.apply_cuts(cuts=self.cuts_signal_region, complement=True)
        self.sidebands = sidebands.features[...,1:].clone()
        sidebands.preprocess()
        self.sidebands_preprocess = sidebands.features.clone()
        self.summary_stats = sidebands.summary_stats
        f.close()

    def get_source_data(self):
        f = h5py.File(self.dataset, 'r') 
        dijets = torch.Tensor(f['jet features'])
        sb1 = PreProcessLHCOlympicsHighLevelData(dijets, methods=self.preprocess_methods, summary_stats=self.summary_stats)
        sb1.apply_cuts(cuts=self.cuts_sideband_low)
        if self.exchange_target_with_source: self.target = sb1.features[...,1:][:self.num_dijets].clone()
        else: 
            self.source_test = sb1.features[...,1:][self.num_dijets:].clone()
            self.source = sb1.features[...,1:][:self.num_dijets].clone()
        sb1.preprocess()
        if self.exchange_target_with_source: self.target_preprocess = sb1.features[:self.num_dijets].clone()
        else: self.source_preprocess = sb1.features[:self.num_dijets].clone()
        f.close()

    def get_target_data(self):
        f = h5py.File(self.dataset, 'r') 
        dijets = torch.Tensor(f['jet features'])
        sb2 = PreProcessLHCOlympicsHighLevelData(dijets,  methods=self.preprocess_methods, summary_stats=self.summary_stats)
        sb2.apply_cuts(cuts=self.cuts_sideband_high)
        if self.exchange_target_with_source: 
            self.source = sb2.features[...,1:][:self.num_dijets].clone()
            self.source_test = sb2.features[...,1:][self.num_dijets:].clone()
        else: self.target = sb2.features[...,1:][:self.num_dijets].clone()
        sb2.preprocess()
        if self.exchange_target_with_source: self.source_preprocess = sb2.features[:self.num_dijets].clone()
        else: self.target_preprocess = sb2.features[:self.num_dijets].clone()
        f.close()

    def get_background_data(self):
        f = h5py.File(self.dataset, 'r') 
        dijets = torch.Tensor(f['jet features'])
        sr = PreProcessLHCOlympicsHighLevelData(dijets,  methods=self.preprocess_methods, summary_stats=self.summary_stats)
        sr.apply_cuts(cuts=self.cuts_signal_region, background=True)
        self.background = sr.features[...,1:].clone()
        sr.preprocess()
        self.background_preprocess = sr.features.clone()
        f.close()



class LHCOlympicsLowLevelDataset(Dataset):

    def __init__(self, configs: dataclass, exchange_target_with_source=False):

        self.dataset = configs.dataset
        self.preprocess_methods = configs.preprocess 
        self.num_dijets = configs.num_dijets
        self.cuts_sideband_low = configs.cuts_sideband_low 
        self.cuts_sideband_high = configs.cuts_sideband_high
        self.cuts_signal_region = {'mjj': [configs.cuts_sideband_low['mjj'][1], configs.cuts_sideband_high['mjj'][0]]}
        self.cuts_sideband_region = {'mjj': [configs.cuts_sideband_low['mjj'][0], configs.cuts_sideband_high['mjj'][1]]}
        self.summary_stats = None
        self.exchange_target_with_source = exchange_target_with_source
    
        ''' data attributes:
            (label, mjj, px_j1, py_j1, pz_j1, e_j1, px_j2, py_j2, pz_j2, e_j2)
            (label, mjj, pt_j1, eta_j1, phi_j1, m_j1, pt_j2, eta_j2, phi_j2, m_j2)
            - target: SB2 data
            - source: SB1 data
        '''
        
        self.get_stats()
        self.get_source_data()
        self.get_target_data()
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

    def get_stats(self):
        f = h5py.File(self.dataset, 'r') 
        dijets = torch.Tensor(f['jet features'])
        sidebands = PreProcessLHCOlympicsLowLevelData(dijets, methods=self.preprocess_methods)
        sidebands.apply_cuts(cuts=self.cuts_sideband_region)
        sidebands.apply_cuts(cuts=self.cuts_signal_region, complement=True)
        self.sidebands = sidebands.features[...,2:].clone()
        sidebands.preprocess()
        self.sidebands_preprocess = sidebands.features.clone()
        self.summary_stats = sidebands.summary_stats
        f.close()

    def get_source_data(self):
        f = h5py.File(self.dataset, 'r') 
        dijets = torch.Tensor(f['jet features'])
        sb1 = PreProcessLHCOlympicsLowLevelData(dijets, methods=self.preprocess_methods, summary_stats=self.summary_stats)
        sb1.apply_cuts(cuts=self.cuts_sideband_low)
        if self.exchange_target_with_source: self.target = sb1.features[...,2:][:self.num_dijets].clone()
        else: 
            self.source_test = sb1.features[...,2:][self.num_dijets:].clone()
            self.source = sb1.features[...,2:][:self.num_dijets].clone()
        sb1.preprocess()
        if self.exchange_target_with_source: self.target_preprocess = sb1.features[:self.num_dijets].clone()
        else: self.source_preprocess = sb1.features[:self.num_dijets].clone()
        f.close()

    def get_target_data(self):
        f = h5py.File(self.dataset, 'r') 
        dijets = torch.Tensor(f['jet features'])
        sb2 = PreProcessLHCOlympicsLowLevelData(dijets,  methods=self.preprocess_methods, summary_stats=self.summary_stats)
        sb2.apply_cuts(cuts=self.cuts_sideband_high)
        if self.exchange_target_with_source: 
            self.source = sb2.features[...,2:][:self.num_dijets].clone()
            self.source_test = sb2.features[...,2:][self.num_dijets:].clone()
        else: self.target = sb2.features[...,2:][:self.num_dijets].clone()
        sb2.preprocess()
        if self.exchange_target_with_source: self.source_preprocess = sb2.features[:self.num_dijets].clone()
        else: self.target_preprocess = sb2.features[:self.num_dijets].clone()
        f.close()

    def get_background_data(self):
        f = h5py.File(self.dataset, 'r') 
        dijets = torch.Tensor(f['jet features'])
        sr = PreProcessLHCOlympicsLowLevelData(dijets,  methods=self.preprocess_methods, summary_stats=self.summary_stats)
        sr.apply_cuts(cuts=self.cuts_signal_region, background=True)
        self.background = sr.features[...,2:].clone()
        sr.preprocess()
        self.background_preprocess = sr.features.clone()
        f.close()
