import torch
import h5py
from torch.utils.data import Dataset
from dataclasses import dataclass
import numpy as np

from DynGenModels.datamodules.cathode.dataprocess import PreProcessCathodeData

class CathodeClassifierDataset(Dataset):

    def __init__(self, configs: dataclass):

        self.model_data = configs.data_gen_model
        self.ref_data = configs.data_reference
        self.snr = configs.signal_noise_ratio
        self.mass_window = configs.mass_window
        self.preprocess_methods = configs.preprocess
        
        '''
            data: preprocessed features (mj1, delta_mj, tau21_1, tau21_2)
            labels: class labels 0: background interpolation, 1: data (s+b)              
            shapes:
                labels : (class labels)
                data : (mj1, delta_mj, tau21_1, tau21_2)
            
         '''
            

        self.get_reference_data()
        self.get_generated_data()

    def __getitem__(self, idx):
        output = {}
        output['SR data'] = self.data_preprocess[idx]
        output['labels'] = self.labels[idx]
        return output

    def __len__(self):
        return self.data_preprocess.size(0)
    
    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def get_generated_data(self):

        '''
            background_estimation : (1, mj1, delta_mj, tau21_1, tau21_2)
            labels : (class labels)
            data : (mj1, delta_mj, tau21_1, tau21_2)
            data_preprocess : (mj1, delta_mj, tau21_1, tau21_2)
        '''
                
        f = h5py.File(self.model_data, 'r') 
        data = torch.Tensor(f['jet high level features'])
        self.background_estimation  = torch.concat([torch.zeros(data.size(0), 1), data[..., 1:]], dim=-1)
        idx = torch.randperm(self.data_ref.size(0))[:self.background_estimation.size(0)]
        self.data_ref = self.data_ref[idx] # downsize to match background sample size
        data = torch.concat([self.background_estimation, self.data_ref], dim=0)
        data = data[torch.randperm(data.size(0))]
        self.labels = data[..., 0].unsqueeze(-1)
        self.data = data[..., 1:]
        _data = PreProcessCathodeData(data=self.data, methods=self.preprocess_methods)
        _data.preprocess()
        self.data_preprocess = _data.features.clone()
        
        f.close()

    def get_reference_data(self):

        '''
            background_truth : (0, mjj,  mj1, delta_mj, tau21_1, tau21_2)
            signal_truth : (1, mjj,  mj1, delta_mj, tau21_1, tau21_2)
            data_ref : (0, mj1, delta_mj, tau21_1, tau21_2)
        '''

        f = h5py.File(self.ref_data, 'r') 
        data = torch.Tensor(f['jet features'])
        mask_mass_window = (data[...,1] > self.mass_window[0] ) & (data[...,1] < self.mass_window[1])
        data = data[mask_mass_window]
        mask_signal = data[...,0] == 1
        signal_truth = data[mask_signal]
        background_truth = data[~mask_signal]

        S = signal_truth.size(0)
        B = background_truth.size(0)
        self.signal_truth, self.background_truth = signal_truth[:S//2], background_truth[:B//2]
        self.signal_truth_test, self.background_truth_test = signal_truth[S//2:], background_truth[B//2:]
        
        S=self.signal_truth.size(0)
        B=self.background_truth.size(0)
        idx = torch.randperm(S)[: int(self.snr * B)]
        self.signal_truth = self.signal_truth[idx]
        print('INFO: truth data S={}, B={}, SNR={}'.format(self.signal_truth.shape[0], B, np.round(self.signal_truth.shape[0]/B, 4)))
        data = torch.cat([self.signal_truth, self.background_truth], dim=0)
        data = data[torch.randperm(data.size(0))]
        self.data_ref = torch.concat([torch.ones(data.size(0), 1), data[..., 2:]], dim=-1)
    
        self.signal_truth_test = torch.concat([torch.ones(self.signal_truth_test.size(0), 1), self.signal_truth_test[..., 2:]], dim=-1)
        self.background_truth_test = torch.concat([torch.zeros(self.background_truth_test.size(0), 1), self.background_truth_test[..., 2:]], dim=-1)
        self.data_ref_test = torch.cat([self.signal_truth_test, self.background_truth_test], dim=0)
        self.data_ref_test = self.data_ref_test[torch.randperm(self.data_ref_test.size(0))]
        self.labels_test = self.data_ref_test[..., 0].unsqueeze(-1)
        self.data_test = self.data_ref_test[..., 1:]
        _data = PreProcessCathodeData(data=self.data_test, methods=self.preprocess_methods)
        _data.preprocess()
        self.data_preprocess_test = _data.features.clone()
        self.data_ref_test = torch.concat([self.labels_test, self.data_preprocess_test], dim=-1)

        f.close()

        # f = h5py.File(self.ref_data, 'r') 
        # data = torch.Tensor(f['jet features'])
        # mask_mass_window = (data[...,1] > self.mass_window[0] ) & (data[...,1] < self.mass_window[1])
        # data = data[mask_mass_window]
        # mask_signal = data[...,0] == 1
        # self.signal_truth = data[mask_signal]
        # self.background_truth = data[~mask_signal]
        # S=self.signal_truth.size(0)
        # B=self.background_truth.size(0)
        # idx = torch.randperm(S)[: int(self.snr * B)]
        # self.signal_truth = self.signal_truth[idx]
        # print('INFO: truth data S={}, B={}, SNR={}'.format(self.signal_truth.shape[0], B, np.round(self.signal_truth.shape[0]/B, 4)))
        # data = torch.cat([self.signal_truth, self.background_truth], dim=0)
        # data = data[torch.randperm(data.size(0))]
        # self.data_ref = torch.concat([torch.zeros(data.size(0), 1), data[..., 2:]], dim=-1)
        # f.close()