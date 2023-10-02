import torch
import numpy as np
from torch.utils.data import Dataset
from dataclasses import dataclass

from DynGenModels.datamodules.toys.dataprocess import PreProcessGaussData

class Gauss2MoonsDataset(Dataset):

    def __init__(self, configs: dataclass):
        
        self.num_points = configs.num_points
        self.N = configs.num_gaussians
        self.gauss_N_scale = configs.gauss_N_scale
        self.gauss_N_var = configs.gauss_N_var 
        self.gauss_centers = configs.gauss_centers
        self.moon_2_noise = configs.moon_2_noise

        ''' datasets:
            source data (x0) :  N gaussians on unit circle
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
        from torchdyn.datasets import generate_moons
        x0, _ = generate_moons(self.num_points, self.moon_2_noise)
        return x0 * 3 - 1

    def get_source_data(self,  dim=2):

            m = torch.distributions.multivariate_normal.MultivariateNormal(
                torch.zeros(dim), np.sqrt(self.gauss_N_var) * torch.eye(dim)
                )

            centers = torch.tensor(self.gauss_centers, dtype=torch.float32) * self.gauss_N_scale
            noise = m.sample((self.num_points,))
            multi = torch.multinomial(torch.ones(self.N), self.num_points, replacement=True)
            data = []

            for i in range(self.num_points):
                data.append(centers[multi[i]] + noise[i])
            return torch.stack(data)

class RectifiedSmearedGaussDataset(torch.utils.data.Dataset):
    def __init__(self, configs: dataclass):
        
        self.num_points = configs.num_points
        self.noise_cov = torch.Tensor(configs.noise_cov)
        self.cuts = configs.cuts
        self.preprocess_methods = configs.preprocess
        self.summary_stats = None


        ''' datasets:
            source data (x0) :  std gaussian
            target data (x1) :  3 perpendicular smeared gaussians
        '''
        self.truth = self.get_truth_data()
        self.target_preprocess = self.get_target_data(self.truth)
        self.source_preprocess = self.get_source_data()

    def __getitem__(self, idx):
        output = {}
        output['target'] = self.target_preprocess[idx]
        output['source'] = self.source_preprocess[idx]
        return output

    def __len__(self):
        return self.source.size(0)
    
    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def get_target_data(self, truth_data):
        noise_dist = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(2), self.noise_cov)
        noise = noise_dist.sample((truth_data.shape[0],))
        noisy_data = truth_data + noise
        target = PreProcessGaussData(noisy_data, cuts=self.cuts, methods=self.preprocess_methods)
        target.apply_cuts()
        self.target = target.features
        target.preprocess()
        self.summary_stats = target.summary_stats
        print("INFO: loading and preprocessing data...")
        print('\t- target dataset shape: {}'.format(target.features.shape))
        return target.features

    def get_source_data(self):
        self.source = torch.randn_like(self.target, dtype=torch.float32)
        print('\t- source dataset: std gaussian \n \t- source shape: {}'.format( self.source.shape))
        return self.source

    def get_truth_data(self, dim=2):
            data_means = torch.Tensor([
                [-2.0, 0.0],
                [0.0, -2.0],
                [0.0, 2.0]
            ])

            data_covars = torch.Tensor([
                [[0.3**2, 0],[0, 1]],
                [[1, 0],[0, 0.3**2]],
                [[1, 0],[0, 0.3**2]]])

            distributions = [torch.distributions.multivariate_normal.MultivariateNormal(mean, covar) for mean, covar in zip(data_means, data_covars)]
            
            multi = torch.multinomial(torch.ones(len(distributions)), self.num_points, replacement=True)
            data = []

            for i in range(self.num_points):
                selected_distribution = distributions[multi[i]]
                sample = selected_distribution.sample()
                data.append(sample)

            return torch.stack(data)

