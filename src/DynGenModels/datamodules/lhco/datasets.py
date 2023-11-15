import torch
import h5py
from torch.utils.data import Dataset
from dataclasses import dataclass

from DynGenModels.datamodules.lhco.dataprocess import PreProcessLHCOlympicsData, PreProcessLHCOlympicsHighLevelData


class LHCOlympicsHighLevelDataset(Dataset):

    def __init__(self, configs: dataclass, exchange_target_with_source=False):

        self.dataset = configs.dataset
        self.preprocess_methods = configs.preprocess 
        self.num_dijets = configs.num_dijets
        self.cuts_sideband_low = configs.cuts_sideband_low 
        self.cuts_sideband_high = configs.cuts_sideband_high
        self.cuts_signal_region = {'mjj': [configs.cuts_sideband_low['mjj'][1], configs.cuts_sideband_high['mjj'][0]]}
        self.summary_stats = None
        self.exchange_target_with_source = exchange_target_with_source
    
        ''' data attributes:
            - target: SB2 data
            - source: SB1 data
        '''
        
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

    def get_source_data(self):
        f = h5py.File(self.dataset, 'r') 
        dijets = torch.Tensor(f['jet features'])
        sb1 = PreProcessLHCOlympicsHighLevelData(dijets, num_dijets=self.num_dijets, cuts=self.cuts_sideband_low, methods=self.preprocess_methods)
        sb1.apply_cuts()
        if self.exchange_target_with_source: self.target = sb1.features.clone()
        else: self.source = sb1.features.clone()
        sb1.preprocess()
        if self.exchange_target_with_source: 
            self.target_preprocess = sb1.features.clone()
            self.summary_stats_target = sb1.summary_stats
        else: 
            self.source_preprocess = sb1.features.clone()
            self.summary_stats_source = sb1.summary_stats
        f.close()

    def get_target_data(self):
        f = h5py.File(self.dataset, 'r') 
        dijets = torch.Tensor(f['jet features'])
        sb2 = PreProcessLHCOlympicsHighLevelData(dijets, num_dijets=self.num_dijets, cuts=self.cuts_sideband_high, methods=self.preprocess_methods)
        sb2.apply_cuts()
        if self.exchange_target_with_source: self.source = sb2.features.clone()
        else: self.target = sb2.features.clone()
        sb2.preprocess()
        if self.exchange_target_with_source: 
            self.source_preprocess = sb2.features.clone()
            self.summary_stats_source = sb2.summary_stats
        else: 
            self.target_preprocess = sb2.features.clone()
            self.summary_stats_target = sb2.summary_stats
        f.close()

    def get_background_data(self):
        f = h5py.File(self.dataset, 'r') 
        dijets = torch.Tensor(f['jet features'])
        sr = PreProcessLHCOlympicsHighLevelData(dijets, num_dijets=self.num_dijets, cuts=self.cuts_signal_region, methods=self.preprocess_methods)
        sr.apply_cuts(background=True)
        self.background = sr.features.clone()
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
        self.summary_stats = None
        self.exchange_target_with_source = exchange_target_with_source
    
        ''' data attributes:
            - target: SB2 data
            - source: SB1 data
        '''
        
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

    def get_source_data(self):
        f = h5py.File(self.dataset, 'r') 
        dijets = torch.Tensor(f['jet features'])
        sb1 = PreProcessLHCOlympicsHighLevelData(dijets, num_dijets=self.num_dijets, cuts=self.cuts_sideband_low, methods=self.preprocess_methods)
        sb1.apply_cuts()
        if self.exchange_target_with_source: self.target = sb1.features.clone()
        else: self.source = sb1.features.clone()
        sb1.preprocess()
        self.summary_stats = sb1.summary_stats
        if self.exchange_target_with_source: self.target_preprocess = sb1.features.clone()
        else: self.source_preprocess = sb1.features.clone()
        f.close()

    def get_target_data(self):
        f = h5py.File(self.dataset, 'r') 
        dijets = torch.Tensor(f['jet features'])
        sb2 = PreProcessLHCOlympicsHighLevelData(dijets, num_dijets=self.num_dijets, cuts=self.cuts_sideband_high, methods=self.preprocess_methods, summary_stats=self.summary_stats)
        sb2.apply_cuts()
        if self.exchange_target_with_source: self.source = sb2.features.clone()
        else: self.target = sb2.features.clone()
        sb2.preprocess()
        if self.exchange_target_with_source: self.source_preprocess = sb2.features.clone()
        else: self.target_preprocess = sb2.features.clone()
        f.close()

    def get_background_data(self):
        f = h5py.File(self.dataset, 'r') 
        dijets = torch.Tensor(f['jet features'])
        sr = PreProcessLHCOlympicsHighLevelData(dijets, num_dijets=self.num_dijets, cuts=self.cuts_signal_region, methods=self.preprocess_methods, summary_stats=self.summary_stats)
        sr.apply_cuts(background=True)
        self.background = sr.features.clone()
        sr.preprocess()
        self.background_preprocess = sr.features.clone()
        f.close()




class LHCOlympicsDataset(Dataset):

    def __init__(self, configs: dataclass):

        self.dim_input = configs.dim_input
        self.feature_list = configs.features
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

        self.get_features()
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
        dijets = torch.Tensor(f['jet features'])[:, self.feat_mask]
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
        dijets = torch.Tensor(f['jet features'])[:, self.feat_mask]
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
        dijets = torch.Tensor(f['jet features'])[:, self.feat_mask]
        sr = PreProcessLHCOlympicsData(dijets, num_dijets=self.num_dijets, cuts=self.cuts_signal_region, methods=self.preprocess_methods)
        sr.apply_cuts(background=True)
        self.background = sr.features.clone()
        self.background_context = sr.log_mjj.clone()
        sr.preprocess()
        self.background_summary_stats = sr.summary_stats
        self.background_preprocess = sr.features.clone()
        f.close()

    def get_features(self):
        jet_features = {'px_0': 0, 'py_0': 1, 'pz_0': 2, 'm_0': 3, 'N_0': 4, 'tau1_0': 5, 'tau2_0': 6, 'tau3_0': 7,
                        'px_1': 8, 'py_1': 9, 'pz_1': 10, 'm_1': 11, 'N_1': 12, 'tau1_1': 13, 'tau2_1': 14, 'tau3_1': 15}
        temp = []                          
        for f in self.feature_list: temp.append(jet_features[f])
        self.feat_mask = torch.ones(self.dim_input, dtype=torch.bool)
        self.feat_mask[temp] = False

