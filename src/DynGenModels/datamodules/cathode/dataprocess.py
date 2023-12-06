from typing import Any
import torch
import numpy as np

class PreProcessCathodeData:
 
    def __init__(self, 
                 data, 
                 methods: list=['standardize'],
                 summary_stats: dict=None
                 ):
        
        self.features = data
        self.methods = methods
        self.summary_stats = {} if summary_stats is None else summary_stats

    def preprocess(self):
        if self.methods is not None:
            for method in self.methods:
                method = getattr(self, method, None)
                if method and callable(method): method()
                else: raise ValueError('Preprocessing method {} not implemented'.format(method))
        else: pass

    def normalize(self, alpha=1e-5):
        """ normalize data 
        """
        if 'min' not in self.summary_stats.keys(): self.summary_stats['min'] = torch.min(self.features, dim=0)[0] + alpha
        if 'max' not in self.summary_stats.keys(): self.summary_stats['max'] = torch.max(self.features, dim=0)[0] - alpha
        self.features = (self.features - self.summary_stats['min'] ) / (self.summary_stats['max'] - self.summary_stats['min'])
    
    def standardize(self, sigma: float=1.0):
        """ standardize data
        """
        if 'mean' not in self.summary_stats.keys(): self.summary_stats['mean'] = torch.mean(self.features, dim=0)
        if 'std' not in self.summary_stats.keys(): self.summary_stats['std'] = torch.std(self.features, dim=0)
        self.features = (self.features - self.summary_stats['mean']) / (self.summary_stats['std'] / sigma)


class PostProcessCathodeData:
 
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
        self.features = self.features * (std / sigma) + mean

    def inverse_normalize(self, alpha=1e-5):
        min = self.summary_stats['min'].to(self.features.device) - alpha
        max = self.summary_stats['max'].to(self.features.device) + alpha
        self.features = (self.features) * (max - min) + min 
    
