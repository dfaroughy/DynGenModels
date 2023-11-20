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
        self.features = torch.tensor(self.features[:self.num_dijets])  
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
        self.features = (self.features - self.summary_stats['min']) / (self.summary_stats['max'] - self.summary_stats['min'])
    
    def logit_transform(self, alpha=1e-5):
        """ smoothen rectified distribution with logit transform
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

    def inverse_normalize(self):
        min = self.summary_stats['min'].to(self.features.device)
        max = self.summary_stats['max'].to(self.features.device)
        self.features = self.features * (max - min) + min
    
    def inverse_logit_transform(self, alpha=1e-5):
        exp = torch.exp(self.features)
        self.features = exp / (1 + exp)
        self.features = (self.features - alpha) / (1 - 2 * alpha)
    





class PreProcessLHCOlympicsData:

    def __init__(self, 
                 data, 
                 num_dijets: int=None,
                 cuts: dict={'mjj': None},
                 methods: list=None
                 ):
        
        self.features = data
        self.cuts = cuts 
        self.num_dijets = num_dijets
        self.methods = methods
        self.summary_stats = {}

    def apply_cuts(self, background=False):
        self.selection_cuts(feature='mjj', cut=self.cuts['mjj'], background=background)

    def preprocess(self):        
        if self.methods is not None:
            for method in self.methods:
                method = getattr(self, method, None)
                if method and callable(method): method()
                else: raise ValueError('Preprocessing method {} not implemented'.format(method))
        else: pass
    
    def selection_cuts(self, feature, cut=None, background=False):
        if cut is None: cut=[-np.inf, np.inf]
        dic={'mjj':1}
        if background:
            self.features = self.features[self.features[...,0]==0]
        mask = (self.features[..., dic[feature]] >= cut[0]) & (self.features[..., dic[feature]] <= cut[1])
        self.features = self.features[mask]
        self.features = self.features[:self.num_dijets]
        self.log_mjj = torch.log(self.features[..., 1])
        self.features = torch.cat([self.features[..., 3:7], self.features[..., 8:12]], dim=-1)  
        
    def log_pt(self):
        self.features[..., 0] = torch.log(self.features[..., 0])
        self.features[..., 4] = torch.log(self.features[..., 4])

    def log_mass(self):
        self.features[..., 3] = torch.log(self.features[..., 3]) + 2e-5
        self.features[..., 7] = torch.log(self.features[..., 7]) + 2e-5


class PostProcessLHCOlympicsData:

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

    def inverse_log_pt(self):
         self.features[..., 0] = torch.exp(self.features[..., 0])
         self.features[..., 4] = torch.exp(self.features[..., 4])

    def inverse_log_mass(self):
         self.features[..., 3] = torch.exp(self.features[..., 3] - 2e-5)
         self.features[..., 7] = torch.exp(self.features[..., 7] - 2e-5)
    