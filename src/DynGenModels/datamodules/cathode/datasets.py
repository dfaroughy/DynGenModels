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
        self.num_samples = configs.num_samples
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
        self.background_estimation = torch.concat([torch.zeros(data.size(0), 1), data[..., 1:]], dim=-1)
        assert self.background_estimation.size(0) >= self.num_samples, 'ERROR: background estimation size {} is smaller than num_samples'.format(self.background_estimation.size(0))
        self.background_estimation = self.background_estimation[:self.num_samples]
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
        
        #...get data inside mass window SR
        
        mask_mass_window = (data[...,1] > self.mass_window[0] ) & (data[...,1] < self.mass_window[1])
        data = data[mask_mass_window]

        #...signal and background truth

        mask_signal = data[...,0] == 1
        signal = data[mask_signal]
        background = data[~mask_signal]

        #...shuffle 

        signal = signal[torch.randperm(signal.size(0))]
        background = background[torch.randperm(background.size(0))]

        print('INFO: total truth data available in SR: S={}, B={}'.format(signal.shape[0], background.shape[0]))

        B = int(self.num_samples // (1 + self.snr))
        S = int(B * self.snr) 

        print('INFO: train/val data in SR = {} = S ({}) + B ({}) at signal-to-noise ratio of {}'.format(S+B, S, B, self.snr))
        print('INFO: train/val generated background in SR = {}'.format(self.num_samples))

        #...get SR data (ref)

        self.signal_truth = signal[:S]
        self.background_truth = background[:B]
        data = torch.cat([self.signal_truth, self.background_truth], dim=0)
        data = data[torch.randperm(data.size(0))]
        self.data_ref = torch.concat([torch.ones(data.size(0), 1), data[..., 2:]], dim=-1)

        #...get test data as complement of data_ref

        self.signal_test = signal[S:]
        self.background_test = background[B:]
        data = torch.cat([self.signal_test , self.background_test ], dim=0)
        data_test = data[torch.randperm(data.size(0))]
        label_truth = data_test[..., 0].unsqueeze(-1)
        data_test = data_test[..., 2:]
        _data = PreProcessCathodeData(data=data_test, methods=self.preprocess_methods)
        _data.preprocess()
        data_test = _data.features.clone()
        self.data_test = torch.concat([label_truth, data_test], dim=-1)
        f.close()
