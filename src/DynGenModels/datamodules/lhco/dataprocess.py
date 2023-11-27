from typing import Any
import torch
import numpy as np


class PreProcessLHCOlympicsHighLevelData:
 
    def __init__(self, 
                 data, 
                 num_dijets: int=None,
                 cuts: dict={'mjj': None},
                 methods: list=None,
                 summary_stats: dict=None
                 ):
        
        self.features = data
        self.cuts = cuts 
        self.num_dijets = num_dijets
        self.methods = methods
        self.summary_stats = {} if summary_stats is None else summary_stats

    def apply_cuts(self, cuts, complement=False, background=False):
        self.selection_cuts(feature='mjj', cuts=cuts, complement=complement, background=background)

    def format(self):
        self.features = self.features[..., 1:] # remove truth label form features

    def preprocess(self, format: bool=True):
        if format: self.format()
        if self.methods is not None:
            for method in self.methods:
                method = getattr(self, method, None)
                if method and callable(method): method()
                else: raise ValueError('Preprocessing method {} not implemented'.format(method))
        else: pass
    
    def selection_cuts(self, feature, cuts=None, complement=False, background=False):
        dic={'mjj':1, 'mj1':2, 'delta_m':3, 'tau21_1':4, 'tau21_2':5}
        if background: self.features = self.features[self.features[...,0]==0]
        mask = (self.features[..., dic[feature]] >= cuts[feature][0]) & (self.features[..., dic[feature]] <= cuts[feature][1])
        if complement: mask = ~mask
        self.features = self.features[mask]

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

    def logit_transform(self, alpha=1e-5):
        """ logit transform data
        """
        self.features = self.features * (1 - 2 * alpha) + alpha
        self.features = torch.log(self.features / (1 - self.features))


class PostProcessLHCOlympicsHighLevelData:

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
    
    def inverse_logit_transform(self, alpha=1e-5):
        exp = torch.exp(self.features)
        self.features = exp / (1 + exp)
        self.features = (self.features - alpha) / (1 - 2 * alpha)
    

class PreProcessLHCOlympicsLowLevelData:
 
    def __init__(self, 
                 data, 
                 num_dijets: int=None,
                 cuts: dict={'mjj': None},
                 methods: list=None,
                 summary_stats: dict=None
                 ):
        
        self.features = data
        self.cuts = cuts 
        self.num_dijets = num_dijets
        self.methods = methods
        self.summary_stats = {} if summary_stats is None else summary_stats

    def apply_cuts(self, cuts, complement=False, background=False):
        self.selection_cuts(feature='mjj', cuts=cuts, complement=complement, background=background)

    def format(self):
        self.features = self.features[..., 2:] # remove truth label form features

    def preprocess(self, format: bool=True):
        if format: self.format()
        if self.methods is not None:
            for method in self.methods:
                method = getattr(self, method, None)
                if method and callable(method): method()
                else: raise ValueError('Preprocessing method {} not implemented'.format(method))
        else: pass
    
    def selection_cuts(self, feature, cuts=None, complement=False, background=False):
        dic={'mjj':1, 'px_j1':2, 'py_j1':3, 'pz_j1':4, 'e_j1':5, 'px_j2':6, 'py_j2':7, 'pz_j2':8, 'e_j2':9}
        if background: self.features = self.features[self.features[...,0]==0]
        mask = (self.features[..., dic[feature]] >= cuts[feature][0]) & (self.features[..., dic[feature]] <= cuts[feature][1])
        if complement: mask = ~mask
        self.features = self.features[mask]

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

    def logit_transform(self, alpha=1e-5):
        """ logit transform data
        """
        self.features = self.features * (1 - 2 * alpha) + alpha
        self.features = torch.log(self.features / (1 - self.features))



class PostProcessLHCOlympicsLowLevelData:

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
    
    def inverse_logit_transform(self, alpha=1e-5):
        exp = torch.exp(self.features)
        self.features = exp / (1 + exp)
        self.features = (self.features - alpha) / (1 - 2 * alpha)