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
        # pt jet 1
        self.features[...,:3] = self.features[...,:3]  * (1 - 2 * alpha) + alpha
        self.features[...,:3]  = torch.log(self.features[...,:3]  / (1 - self.features[...,:3] ))


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
        exp = torch.exp(self.features[...,:3] )
        self.features[...,:3]  = exp / (1 + exp)
        self.features[...,:3]  = (self.features[...,:3]  - alpha) / (1 - 2 * alpha)
    

class PreProcessLHCOlympicsLowLevelData:
 
    def __init__(self, 
                 data, 
                 num_dijets: int=None,
                 cuts: dict={'mjj': None},
                 methods: list=None,
                 summary_stats: dict=None,
                 coords = 'px_py_pz_e'
                 ):
        
        self.features = data
        self.cuts = cuts 
        self.num_dijets = num_dijets
        self.methods = methods
        self.coords = coords
        self.summary_stats = {} if summary_stats is None else summary_stats

    def apply_cuts(self, cuts, complement=False, background=False):
        self.selection_cuts(feature='mjj', cuts=cuts, complement=complement, background=background)
        if self.coords == 'pt_eta_phi_m':
            self.features[..., 4][self.features[..., 4] < 0] += 2 * np.pi  # phi 0...2pi, jet 1
            self.features[..., 8][self.features[..., 8] < 0] += 2 * np.pi # phi 0...2pi, jet 2

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
        dic={'mjj':1}
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
    
    def standardize(self, sigma: float=1.0, mu: float=0.0):
        """ standardize data
        """
        if 'mean' not in self.summary_stats.keys(): self.summary_stats['mean'] = torch.mean(self.features, dim=0)
        if 'std' not in self.summary_stats.keys(): self.summary_stats['std'] = torch.std(self.features, dim=0)
        self.features = mu + (self.features - self.summary_stats['mean']) / (self.summary_stats['std'] / sigma)

    def logit_transform(self, alpha=1e-5):
        """ logit transform data
        """
        # pt jet 1
        self.features[..., 0] = self.features[..., 0] * (1 - 2 * alpha) + alpha
        self.features[..., 0] = torch.log(self.features[..., 0] / (1 - self.features[..., 0]))
        
        # phi jet 1
        self.features[..., 2] = self.features[..., 2] * (1 - 2 * alpha) + alpha
        self.features[..., 2] = torch.log(self.features[..., 2] / (1 - self.features[..., 2]))
        
        # phi jet 2
        self.features[..., 6] = self.features[..., 6] * (1 - 2 * alpha) + alpha
        self.features[..., 6] = torch.log(self.features[..., 6] / (1 - self.features[..., 6]))

    def cosine_transform(self):
        """ cosine transform data
        """
        self.features[..., 2] = torch.cos(self.features[..., 2])
        self.features[..., 6] = torch.cos(self.features[..., 6])

    def log_energy(self, alpha=100):
        """ logit transform data

        """

        self.features[..., 0] =             torch.log(self.features[..., 0] + alpha)
        self.features[..., 3] = torch.log(self.features[..., 3] + alpha)
        self.features[..., 4] = torch.log(self.features[..., 4] + alpha)
        self.features[..., -1] = torch.log(self.features[..., -1] + alpha)


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

    def inverse_standardize(self,  sigma: float=1.0, mu: float=0.0):
        std = self.summary_stats['std'].to(self.features.device)
        mean = self.summary_stats['mean'].to(self.features.device)
        self.features = (self.features - mu) * (std / sigma) + mean

    def inverse_normalize(self, alpha=1e-5):
        min = self.summary_stats['min'].to(self.features.device) - alpha
        max = self.summary_stats['max'].to(self.features.device) + alpha
        self.features = (self.features) * (max - min) + min 
    
    def inverse_logit_transform(self, alpha=1e-5):

        # pt jet 1
        exp = torch.exp(self.features[..., 0])
        self.features[..., 0] = exp / (1 + exp)
        self.features[..., 0] = (self.features[..., 0] - alpha) / (1 - 2 * alpha)

        # phi jet 1
        exp = torch.exp(self.features[..., 2])
        self.features[..., 2] = exp / (1 + exp)
        self.features[..., 2] = (self.features[..., 2] - alpha) / (1 - 2 * alpha)

        # phi jet 2
        exp = torch.exp(self.features[..., 6])
        self.features[..., 6] = exp / (1 + exp)
        self.features[..., 6] = (self.features[..., 6] - alpha) / (1 - 2 * alpha)

    def inverse_cosine_transform(self):
        self.features[..., 2] = torch.acos(self.features[..., 2])
        self.features[..., 6] = torch.acos(self.features[..., 6])
    

    def inverse_log_energy(self, alpha=100):
        """ logit transform data
        """
        self.features[..., 0] = torch.exp(self.features[..., 0]) - alpha
        self.features[..., 3] = torch.exp(self.features[..., 3]) - alpha        
        self.features[..., 4] = torch.exp(self.features[..., 4]) - alpha
        self.features[..., -1] = torch.exp(self.features[..., -1]) - alpha