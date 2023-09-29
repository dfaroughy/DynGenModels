from typing import Any
import torch
import numpy as np

class PreProcessFermiData:

    def __init__(self, 
                 data, 
                 cuts: dict=None,
                 methods: list=['standardize']
                 ):
        
        self.galactic_features = data
        self.cuts = cuts if cuts is not None else {'x': None, 'y': None}
        self.methods = methods
        self.summary_stats = {}

    def apply_cuts(self):
        self.selection_cuts(feature='x', cut=self.cuts['x'])
        self.selection_cuts(feature='y', cut=self.cuts['y'])

    def preprocess(self):        
        #...preprocess with provided methods
        for method in self.methods:
            method = getattr(self, method, None)
            if method and callable(method): method()
            else: raise ValueError('Preprocessing method {} not implemented'.format(method))
    
    def selection_cuts(self, feature, cut=None):
        if cut is None: cut=[-np.inf, np.inf]
        dic={'x':0, 'y':1}
        mask = (self.features[..., dic[feature]] >= cut[0]) & (self.features[..., dic[feature]] <= cut[1])
        self.features = self.features[mask]

    def standardize(self,  sigma: float=1.0):
        """ standardize data to have zero mean and unit variance
        """
        self.summary_stats['mean'] = torch.mean(self.galactic_features, dim=0)
        self.summary_stats['std'] = torch.std(self.galactic_features, dim=0)
        self.galactic_features = (self.galactic_features - self.summary_stats['mean']) * (sigma / self.summary_stats['std'])

    def normalize(self):
        """ normalize data to unit interval
        """
        self.summary_stats['min'], _ = torch.min(self.galactic_features, dim=0)
        self.summary_stats['max'], _ = torch.max(self.galactic_features, dim=0)
        self.galactic_features = (self.galactic_features - self.summary_stats['min']) / ( self.summary_stats['max'] - self.summary_stats['min'])
    
    def logit_transform(self, alpha=1e-5):
        """ smoothen rectified distribution with logit transform
        """
        self.galactic_features = self.galactic_features * (1 - 2 * alpha) + alpha
        self.galactic_features = torch.log(self.galactic_features / (1 - self.galactic_features))



class PostProcessFermiData:

    def __init__(self, 
                 data, 
                 summary_stats,
                 methods: list=['inverse_standardize']
                 ):
        
        self.galactic_features = data
        self.summary_stats = summary_stats
        self.methods = methods

    def postprocess(self):
        for method in self.methods:
            method = getattr(self, method, None)
            if method and callable(method): method()
            else: raise ValueError('Postprocessing method {} not implemented'.format(method))

    def inverse_standardize(self,  sigma: float=1.0):
        self.galactic_features = self.galactic_features * (self.summary_stats['std'] / sigma) + self.summary_stats['mean']

    def inverse_normalize(self):
        self.galactic_features = self.galactic_features * (self.summary_stats['max'] - self.summary_stats['min']) + self.summary_stats['min']
    
    def inverse_logit_transform(self, alpha=1e-5):
        exp = torch.exp(self.galactic_features)
        self.galactic_features = exp / (1 + exp)
        self.galactic_features = (self.galactic_features - alpha) / (1 - 2 * alpha)
