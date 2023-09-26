import torch
import numpy as np
from torch.utils.data import Dataset
from dataclasses import dataclass

class ToysDataset(Dataset):

    def __init__(self, config: dataclass):
        
        self.num_samples = config.num_samples
        self.gaussian_scale = config.gaussian_scale 
        self.gaussian_var = config.gaussian_var 
        self.moon_noise = config.moon_noise

        ''' datasets:
            source data (x0) :  8 gaussians
            target data (x1) :  2 mooons
        '''
        self.target = self.sample_2_moons()
        self.source = self.sample_8_gauss()

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

    def sample_8_gauss(self,  dim=2):

        m = torch.distributions.multivariate_normal.MultivariateNormal(
            torch.zeros(dim), np.sqrt(self.gaussian_var) * torch.eye(dim)
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

        centers = torch.tensor(centers, dtype=torch.float32) * self.gaussian_scale
        noise = m.sample((self.num_samples,))
        multi = torch.multinomial(torch.ones(8), self.num_samples, replacement=True)
        data = []

        for i in range(self.num_samples):
            data.append(centers[multi[i]] + noise[i])
        return torch.stack(data)

    def sample_2_moons(self):
        from sklearn import datasets
        data , _ = datasets.make_moons(n_samples=self.num_samples, noise=self.moon_noise)
        return 3 * torch.tensor(data, dtype=torch.float32) - 1
    

