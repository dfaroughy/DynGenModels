import torch
import numpy as np
from torch.utils.data import Dataset

class ToysDataset(Dataset):

    def __init__(self, 
                 num_samples: int=10000          
                 ):
        
        self.num_samples = num_samples

        ''' datasets:
            source data (x0) :  8 gaussians
            target data (x1) :  2 mooons
        '''

        self.source = self.sample_8_gauss()
        self.target = self.sample_2_moons()

    def __getitem__(self, idx):
        output = {}
        output['source'] = self.source[idx]
        output['target'] = self.target[idx]
        return output

    def __len__(self):
        return self.target.size(0)
    
    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def sample_8_gauss(self,  dim=2, scale=2, var=0.1):

        m = torch.distributions.multivariate_normal.MultivariateNormal(
            torch.zeros(dim), np.sqrt(var) * torch.eye(dim)
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

        centers = torch.tensor(centers) * scale
        noise = m.sample((self.num_samples,))
        multi = torch.multinomial(torch.ones(8), self.num_samples, replacement=True)
        data = []

        for i in range(self.num_samples):
            data.append(centers[multi[i]] + noise[i])
        return torch.stack(data)

    def sample_2_moons(self,  noise=0.2):
        from sklearn import datasets
        data , _ = datasets.make_moons(n_samples=self.num_samples, noise=noise)
        return torch.tensor(data)
    

