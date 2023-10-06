import torch
import numpy as np
from torch.utils.data import Dataset
from dataclasses import dataclass

from DynGenModels.datamodules.toys.dataprocess import PreProcessGaussData

class Gauss_2_Moons_Dataset(Dataset):

    def __init__(self, configs: dataclass):
        
        self.num_points = configs.num_points
        self.N = configs.num_gaussians
        self.gauss_N_scale = configs.gauss_N_scale
        self.gauss_N_var = configs.gauss_N_var 
        self.gauss_centers = configs.gauss_centers
        self.moon_2_noise = configs.moon_2_noise
        self.exchange_source_with_target = configs.exchange_source_with_target

        ''' datasets:
            source data (x0) :  N gaussians on unit circle
            target data (x1) :  2 mooons
        '''

        self.get_target_data()
        self.get_source_data()

    def __getitem__(self, idx):
        output = {}
        output['target'] = self.target[idx] if self.exchange_source_with_target is False else self.source[idx]
        output['source'] = self.source[idx] if self.exchange_source_with_target is False else self.target[idx]
        return output

    def __len__(self):
        return self.target.size(0)
    
    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def get_target_data(self):
        from torchdyn.datasets import generate_moons
        x, _ = generate_moons(self.num_points, self.moon_2_noise)
        self.target = x * 3 - 1

    def get_source_data(self,  dim=2):

        m = torch.distributions.multivariate_normal.MultivariateNormal(
            torch.zeros(dim), np.sqrt(self.gauss_N_var) * torch.eye(dim))

        centers = torch.tensor(self.gauss_centers, dtype=torch.float32) * self.gauss_N_scale
        noise = m.sample((self.num_points,))
        multi = torch.multinomial(torch.ones(self.N), self.num_points, replacement=True)
        data = []

        for i in range(self.num_points):
            data.append(centers[multi[i]] + noise[i])
        
        self.source = torch.stack(data)
