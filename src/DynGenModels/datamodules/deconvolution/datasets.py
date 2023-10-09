import torch
import numpy as np
from torch.utils.data import Dataset
from dataclasses import dataclass

class SmearedGaussDataset(torch.utils.data.Dataset):

    ''' Creates 3 perpendicular gaussians in the 2D plane, 
        applies gaussian smearing with a given noise covariance. 

        output items:
            - smeared: 3 gaussians
            - covariance: noise covariance
    '''
        
    def __init__(self, configs: dataclass):
        
        self.num_points = configs.num_points
        self.noise_cov = torch.Tensor(configs.noise_cov)

        ''' datasets:
        '''
        self.truth = self.get_truth_data()
        self.smeared = self.get_smeared_data(self.truth)
        self.covs = self.get_cov_data()

    def __getitem__(self, idx):
        output = {}
        output['smeared'] = self.smeared[idx]
        output['covariance'] = self.covs[idx]
        output['mask'] = torch.ones_like(self.smeared[idx][..., 0])
        output['context'] = torch.empty_like(self.smeared[idx][..., 0])
        return output

    def __len__(self):
        return self.smeared.size(0)
    
    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def get_smeared_data(self, truth_data):
        noise_dist = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(2), self.noise_cov)
        noise = noise_dist.sample((truth_data.shape[0],))
        return truth_data + noise

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

from DynGenModels.datamodules.deconvolution.dataprocess import PreProcessGaussData

class RectifiedSmearedGaussDataset(torch.utils.data.Dataset):

    ''' Creates 3 perpendicular gaussians in the 2D plane, 
        applies gaussian smearing with a given noise covariance. 
        The Gussian tails are then truncated with provedided cuts. 
        Output are 3 "rectified" Gaussians.

        output items:
            - smeared: 3 rectified gaussians
            - covariance: noise covariance
            - summary_stats: summary statistics of smeared recrtified data
    '''

    def __init__(self, configs: dataclass):
        
        self.num_points = configs.num_points
        self.noise_cov = torch.Tensor(configs.noise_cov)
        self.cuts = configs.cuts
        self.preprocess_methods = configs.preprocess
        self.summary_stats = None
        ''' datasets:
        '''
        self.truth = self.get_truth_data()
        self.smeared_preprocess = self.get_smeared_data(self.truth)
        self.covs = self.get_cov_data()

    def __getitem__(self, idx):
        output = {}
        output['smeared'] = self.smeared_preprocess[idx]
        output['covariance'] = self.covs[idx]
        output['summary_stats'] = self.summary_stats
        return output

    def __len__(self):
        return self.smeared.size(0)
    
    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def get_smeared_data(self, truth_data):
        noise_dist = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(2), self.noise_cov)
        noise = noise_dist.sample((truth_data.shape[0],))
        smeared = PreProcessGaussData(truth_data + noise, cuts=self.cuts, methods=['normalize', 'logit_transform', 'standardize'])
        smeared.apply_cuts()
        self.smeared = smeared.features
        smeared.preprocess()
        self.summary_stats = smeared.summary_stats
        return smeared.features

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
        data = PreProcessGaussData(torch.stack(data), cuts=self.cuts)
        data.apply_cuts()
        
        return data.features

