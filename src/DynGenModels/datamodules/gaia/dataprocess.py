from typing import Any
import torch
import numpy as np

class PreProcessGaiaData:

    def __init__(self, 
                 data, 
                 cuts: dict={'r': None},
                 sun: list=[8.122, 0.0, 0.0208],
                 methods: list=None
                 ):
        
        self.features = data
        self.cuts = cuts 
        self.sun = torch.Tensor(sun)
        self.methods = methods
        self.summary_stats = {}

    def apply_cuts(self):
        ''' remove all stars with distance to sun farther than 'r' kiloparsecs
        '''
        r = torch.norm(self.features[..., :3] - self.sun, dim=-1)
        mask = r <= self.cuts['r'][1]
        self.features = self.features[mask]

    def preprocess(self):        
        if self.methods is not None:
            for method in self.methods:
                method = getattr(self, method, None)
                if method and callable(method): method()
                else: raise ValueError('Preprocessing method {} not implemented'.format(method))
        else: pass
    
    def standardize(self,  sigma: float=1.0):
        """ standardize data to have zero mean and unit variance
        """
        self.summary_stats['mean'] = torch.mean(self.features, dim=0)
        self.summary_stats['std'] = torch.std(self.features, dim=0)
        self.features = (self.features - self.summary_stats['mean']) * (sigma / self.summary_stats['std'])

    def unit_ball_transform(self, c=1+1e-6):
        """ transform data do unit ball around galactic origin
        """
        r = torch.norm(self.features[..., :3] - self.sun, dim=-1)
        self.summary_stats['r_max'] = torch.max(r) * c
        self.features[..., :3] = (self.features[..., :3] - self.sun) / self.summary_stats['r_max']

    def radial_blowup(self):
        norm = torch.linalg.norm(self.features[..., :3], dim=-1, keepdims=True)
        self.features[...,:3] = (self.features[..., :3] / norm) * torch.atanh(norm)
        return self

class PostProcessGaiaData:

    def __init__(self, 
                 data, 
                 summary_stats,
                 sun: list=[8.122, 0.0, 0.0208],
                 methods: list=None
                 ):
        
        self.features = data
        self.sun = torch.Tensor(sun)
        self.summary_stats = summary_stats
        self.methods = methods

    def postprocess(self):
        if self.methods is not None:
            for method in self.methods:
                method = getattr(self, method, None)
                if method and callable(method): method()
                else: raise ValueError('Postprocessing method {} not implemented'.format(method))
        else: pass

    def inverse_standardize(self,  sigma: float=1.0):
        std = self.summary_stats['std'].to(self.features.device)
        mean = self.summary_stats['mean'].to(self.features.device)
        self.features = self.features * (std / sigma) + mean
    
    
    def inverse_unit_ball_transform(self, c=1+1e-6):
        r_max = self.summary_stats['r_max'].to(self.features.device)
        r_sun = self.sun.to(self.features.device)
        self.features[..., :3] = self.features[..., :3] * r_max + r_sun

    def inverse_radial_blowup(self):
        norm = torch.linalg.norm(self.features[..., :3], dim=-1, keepdims=True)
        self.features[...,:3] = (self.features[..., :3] / norm) * torch.tanh(norm)