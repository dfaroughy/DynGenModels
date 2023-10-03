from typing import Any
import torch
import numpy as np

class PreProcessGaiaData:

    def __init__(self, 
                 data, 
                 cuts: dict={'r': None},
                 r_sun: list=[8.122, 0.0, 0.0208],
                 methods: list=['standardize']
                 ):
        
        self.features = data
        self.cuts = cuts 
        self.sun = r_sun
        self.methods = methods
        self.summary_stats = {}

    def apply_cuts(self):
        ''' center sun in origin and apply radial cut at a few kiloparsec
        '''
        x = self.features[..., 0] - self.sun[0]
        y = self.features[..., 1] - self.sun[1]
        z = self.features[..., 2] - self.sun[2]
        r = torch.sqrt(x**2 + y**2 + z**2)
        mask = (r >= self.cuts['radial') & (r <= self.cuts['radial')
        self.features = self.features[mask]

    def preprocess(self):        
        #...preprocess with provided methods
        for method in self.methods:
            method = getattr(self, method, None)
            if method and callable(method): method()
            else: raise ValueError('Preprocessing method {} not implemented'.format(method))
    

    def standardize(self,  sigma: float=1.0):
        """ standardize data to have zero mean and unit variance
        """
        self.summary_stats['mean'] = torch.mean(self.features, dim=0)
        self.summary_stats['std'] = torch.std(self.features, dim=0)
        self.features = (self.features - self.summary_stats['mean']) * (sigma / self.summary_stats['std'])

    def normalize(self):
        """ normalize data to unit interval
        """
        self.summary_stats['min'], _ = torch.min(self.features[..., :3], dim=0)
        self.summary_stats['max'], _ = torch.max(self.features[..., :3], dim=0)
        self.features[..., :3] = (self.features[..., :3] - self.summary_stats['min']) / ( self.summary_stats['max'] - self.summary_stats['min'])
    
    def logit_transform(self, alpha=1e-5):
        """ smoothen rectified distribution with logit transform
        """
        self.features[..., :3] = self.features[..., :3] * (1 - 2 * alpha) + alpha
        self.features[..., :3] = torch.log(self.features[..., :3] / (1 - self.features[..., :3]))



class PostProcessGaiaData:

    def __init__(self, 
                 data, 
                 summary_stats,
                 methods: list=['inverse_standardize']
                 ):
        
        self.features = data
        self.summary_stats = summary_stats
        self.methods = methods

    def postprocess(self):
        for method in self.methods:
            method = getattr(self, method, None)
            if method and callable(method): method()
            else: raise ValueError('Postprocessing method {} not implemented'.format(method))

    def inverse_standardize(self,  sigma: float=1.0):
        self.features = self.features * (self.summary_stats['std'] / sigma) + self.summary_stats['mean']

    def inverse_normalize(self):
        self.features[..., :3] = self.features[..., :3] * (self.summary_stats['max'] - self.summary_stats['min']) + self.summary_stats['min']
    
    def inverse_logit_transform(self, alpha=1e-5):
        exp = torch.exp(self.features)
        self.features[..., :3] = exp / (1 + exp)
        self.features[..., :3] = (self.features[..., :3] - alpha) / (1 - 2 * alpha)
