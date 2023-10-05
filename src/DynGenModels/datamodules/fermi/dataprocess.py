from typing import Any
import torch
import numpy as np

class PreProcessFermiData:

    def __init__(self, 
                 data, 
                 cuts: dict={'theta': None, 'phi': None, 'energy': None},
                 methods: list=None
                 ):
        
        self.features = data
        self.cuts = cuts 
        self.methods = methods
        self.summary_stats = {}

    def apply_cuts(self):
        self.selection_cuts(feature='energy', cut=self.cuts['energy'])
        self.selection_cuts(feature='theta', cut=self.cuts['theta'])
        self.selection_cuts(feature='phi', cut=self.cuts['phi'])

    def preprocess(self):        
        if self.methods is not None:
            for method in self.methods:
                method = getattr(self, method, None)
                if method and callable(method): method()
                else: raise ValueError('Preprocessing method {} not implemented'.format(method))
        else: pass
    
    def selection_cuts(self, feature, cut=None):
        if cut is None: cut=[-np.inf, np.inf]
        dic={'theta':0, 'phi':1, 'energy':2}
        mask = (self.features[..., dic[feature]] >= cut[0]) & (self.features[..., dic[feature]] <= cut[1])
        self.features = self.features[mask]

    def standardize(self,  sigma: float=1.0):
        """ standardize data to have zero mean and unit variance
        """
        self.summary_stats['mean'] = torch.mean(self.features, dim=0)
        self.summary_stats['std'] = torch.std(self.features, dim=0)
        self.features = (self.features - self.summary_stats['mean']) * (sigma / self.summary_stats['std'])

    def normalize(self):
        """ normalize data to unit interval
        """
        self.summary_stats['min'], _ = torch.min(self.features, dim=0)
        self.summary_stats['max'], _ = torch.max(self.features, dim=0)
        self.features = (self.features - self.summary_stats['min']) / ( self.summary_stats['max'] - self.summary_stats['min'])
    
    def logit_transform(self, alpha=1e-5):
        """ smoothen rectified distribution with logit transform
        """
        self.features = self.features * (1 - 2 * alpha) + alpha
        self.features = torch.log(self.features / (1 - self.features))


class PostProcessFermiData:

    def __init__(self, 
                 data, 
                 summary_stats,
                 methods: list=None
                 ):
        
        self.features = data
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
        self.features = self.features * (self.summary_stats['std'] / sigma) + self.summary_stats['mean']

    def inverse_normalize(self):
        self.features = self.features * (self.summary_stats['max'] - self.summary_stats['min']) + self.summary_stats['min']
    
    def inverse_logit_transform(self, alpha=1e-5):
        exp = torch.exp(self.features)
        self.features = exp / (1 + exp)
        self.features = (self.features - alpha) / (1 - 2 * alpha)
