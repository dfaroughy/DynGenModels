import torch
import numpy as np
from torch.utils.data import Dataset
from dataclasses import dataclass

class Gauss2MoonsDataset(Dataset):

    def __init__(self, configs: dataclass):
        
        self.num_points = configs.num_points
        self.gauss_8_scale = configs.gauss_8_scale
        self.gauss_8_var = configs.gauss_8_var 
        self.moon_2_noise = configs.moon_2_noise

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
        data , _ = datasets.make_moons(n_samples=self.num_points, noise=self.moon_2_noise)
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
            noise = m.sample((self.num_points,))
            multi = torch.multinomial(torch.ones(8), self.num_points, replacement=True)
            data = []

            for i in range(self.num_points):
                data.append(centers[multi[i]] + noise[i])
            return torch.stack(data)

class SmearedGaussDataset(torch.utils.data.Dataset):
    def __init__(self, config: dataclass):
        
        self.num_points = config.num_points
        self.noise_cov = torch.Tensor(config.noise_cov)

        ''' datasets:
            source data (x0) :  3 perpendicular smeared gaussians
            target data (x1) :  1 gaussian with smeareing covariance 
        '''
        self.truth = self.get_truth_data()
        self.source = self.get_source_data(self.truth)
        self.covs = self.get_cov_data()

    def __getitem__(self, idx):
        output = {}
        output['source'] = self.source[idx]
        output['covariance'] = self.covs[idx]
        return output

    def __len__(self):
        return self.source.size(0)
    
    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def get_source_data(self, truth_data):
        noise_dist = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(2), self.noise_cov)
        noise = noise_dist.sample((truth_data.shape[0],))
        noisy_data = truth_data + noise
        return noisy_data

    def get_cov_data(self):
        cov = self.noise_cov.unsqueeze(0).repeat(self.num_points, 1, 1)
        return cov

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


class SmearedRectifiedGaussDataset(torch.utils.data.Dataset):
    def __init__(self, config: dataclass):
        
        self.num_points = config.num_points
        self.noise_cov = torch.Tensor(config.noise_cov)
        self.cuts = configs.cuts
        self.preprocess_methods = configs.preprocess 
        self.summary_stats = None

        ''' datasets:
            source data (x0) :  3 perpendicular smeared gaussians
            target data (x1) :  1 gaussian with smeareing covariance 
        '''
        self.truth = self.get_truth_data()
        self.source = self.get_source_data(self.truth)
        self.covs = self.get_cov_data()

    def __getitem__(self, idx):
        output = {}
        output['source'] = self.source[idx]
        output['covariance'] = self.covs[idx]
        return output

    def __len__(self):
        return self.source.size(0)
    
    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def get_source_data(self, truth_data):
        noise_dist = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(2), self.noise_cov)
        noise = noise_dist.sample((truth_data.shape[0],))
        noisy_data = truth_data + noise
        return noisy_data

    def get_cov_data(self):
        cov = self.noise_cov.unsqueeze(0).repeat(self.num_points, 1, 1)
        return cov

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

            data = torch.stack(data)

            #... apply cuts and preprocess:

            data = PreProcessToyData(data, cuts=self.cuts, methods=self.preprocess_methods)
            data.apply_cuts()

            return 
