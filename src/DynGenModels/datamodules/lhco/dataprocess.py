from typing import Any
import torch
import numpy as np

class PreProcessJetNetData:

    def __init__(self, 
                 particle_features,
                 jet_features,
                 mask, 
                 cuts: dict={'num_constituents': None},
                 methods: list=None
                 ):
        
        self.features = particle_features
        self.context = jet_features
        self.mask = mask[..., None]
        self.methods = methods
        self.cuts = cuts
        self.summary_stats = {}

    def apply_cuts(self):
        if self.cuts['num_constituents'] is not None:
            mask = self.mask.sum(dim=1).squeeze() == self.cuts['num_constituents']
            self.features = self.features[mask]
            self.context = self.context[mask]
            self.mask = self.mask[mask]
        else: pass

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
        self.summary_stats['mean'] = (self.features * self.mask).view(-1, self.features.shape[-1]).mean(dim=0)
        self.summary_stats['std'] = (self.features * self.mask).view(-1, self.features.shape[-1]).std(dim=0)
        self.features = (self.features - self.summary_stats['mean']) * (sigma / self.summary_stats['std'])
        self.features = self.features * self.mask


class PostProcessJetNetData:

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
        std = self.summary_stats['std'].to(self.features.device)
        mean = self.summary_stats['mean'].to(self.features.device)
        self.features = self.features * (std  / sigma) + mean
    
