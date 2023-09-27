import torch
import numpy as np
from torch.utils.data import Dataset
from dataclasses import dataclass

class ToysDataset(Dataset):

    def __init__(self, config: dataclass):
        
        self.num_samples = config.num_samples
        self.gauss_8_scale = config.gauss_8_scale
        self.gauss_8_var = config.gauss_8_var 
        self.moon_2_noise = config.moon_2_noise

        ''' datasets:
            source data (x0) :  8 gaussians
            target data (x1) :  2 mooons
        '''
        self.target = self.get_target_data()
        self.source = self.get_source_data()

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

    def get_target_data(self):
        from sklearn import datasets
        data , _ = datasets.make_moons(n_samples=self.num_samples, noise=self.moon_2_noise)
        return 3 * torch.tensor(data, dtype=torch.float32) - 1

    def get_source_data(self,  dim=2):

            m = torch.distributions.multivariate_normal.MultivariateNormal(
                torch.zeros(dim), np.sqrt(self.gauss_8_var) * torch.eye(dim)
                )

            centers = [
                (1, 0),
                (-1, 0),
                (0, 1),
                (0, -1),
                (1.0 / np.sqrt(2), 1.0 / np.sqrt(2)),
                (1.0 / np.sqrt(2), -1.0 / np.sqrt(2)),
                (-1.0 / np.sqrt(2), 1.0 / np.sqrt(2)),
                (-1.0 / np.sqrt(2), -1.0 / np.sqrt(2)),
                ]

            centers = torch.tensor(centers, dtype=torch.float32) * self.gauss_8_scale
            noise = m.sample((self.num_samples,))
            multi = torch.multinomial(torch.ones(8), self.num_samples, replacement=True)
            data = []

            for i in range(self.num_samples):
                data.append(centers[multi[i]] + noise[i])
            return torch.stack(data)

