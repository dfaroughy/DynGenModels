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
        self.cuts = cuts if cuts is not None else {'theta': None, 'phi': None, 'energy': None}
        self.methods = methods
        self.summary_stats = {}

    def preprocess(self):
        #...apply data selection cuts
        self.selection_cuts(feature='energy', cut=self.cuts['energy'])
        self.selection_cuts(feature='theta', cut=self.cuts['theta'])
        self.selection_cuts(feature='phi', cut=self.cuts['phi'])
        
        #...preprocess with provided methods
        for method in self.methods:
            method = getattr(self, method, None)
            if method and callable(method): method()
            else: raise ValueError('Preprocessing method {} not implemented'.format(method))
    
    def selection_cuts(self, feature, cut=None):
        if cut is None: cut=[-np.inf, np.inf]
        dic={'theta':0, 'phi':1, 'energy':2}
        mask = (self.galactic_features[..., dic[feature]] >= cut[0]) & (self.galactic_features[..., dic[feature]] <= cut[1])
        self.galactic_features = self.galactic_features[mask]

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




# class FormatData:

#     def __init__(self, 
#                  data: torch.Tensor=None,
#                  cuts: dict=None):
                 
#         self.data = data
#         self.cuts = cuts if cuts is not None else {'theta': None, 'phi': None, 'energy': None}

#     def selection_cuts(self, feature, cut=None):
#         if cut is None: cut=[-np.inf, np.inf]
#         dic={'theta':0, 'phi':1, 'energy':2}
#         mask = (self.data[..., dic[feature]] >= cut[0]) & (self.data[..., dic[feature]] <= cut[1])
#         self.data = self.data[mask]
#         return self

#     def format(self):
#         self.selection_cuts(feature='energy', cut=self.cuts['energy'])
#         self.selection_cuts(feature='theta', cut=self.cuts['theta'])
#         self.selection_cuts(feature='phi', cut=self.cuts['phi'])

# class PreProcessData:

#     def __init__(self, 
#                  data, 
#                  stats,
#                  methods: list=['standardize']
#                  ):
        
#         self.galactic_features = data
#         self.mean, self.std, self.min, self.max = stats 
#         self.methods = methods

#     def preprocess(self):
#         for method in self.methods:
#             method = getattr(self, method, None)
#             if method and callable(method): method()
#             else: raise ValueError('Preprocessing method {} not implemented'.format(method))
        
#     def standardize(self,  sigma: float=1.0):
#         self.mean = torch.mean(self.galactic_features, dim=0)
#         self.std = torch.std(self.galactic_features, dim=0)
#         self.galactic_features = (self.galactic_features - self.mean) * (sigma / self.std )

#     def normalize(self):
#         self.min, _ = torch.min(data, dim=0)
#         self.max, _ = torch.max(data, dim=0)
#         self.galactic_features = (self.galactic_features - self.min) / ( self.max - self.min )
    
#     def logit_transform(self, alpha=1e-6):
#         self.galactic_features = self.galactic_features * (1 - 2 * alpha) + alpha
#         self.galactic_features = torch.log(self.galactic_features / (1 - self.galactic_features))

#     def save_stats(self):
#         return (self.mean, self.std, self.min, self.max)
    

# class PostProcessData:

#     def __init__(self, 
#                  data, 
#                  stats,
#                  methods: list=['inverse_standardize']
#                  ):
        
#         self.galactic_features = data
#         self.mean, self.std, self.min, self.max = stats 
#         self.methods = methods

#     def postprocess(self):
#         for method in self.methods:
#             method = getattr(self, method, None)
#             if method and callable(method): method()
#             else: raise ValueError('Preprocessing method {} not implemented'.format(method))
        
#     def inverse_standardize(self,  sigma: float=1.0):
#         self.galactic_features = self.galactic_features * (self.std / sigma) + self.mean

#     def inverse_normalize(self):
#         self.galactic_features = self.galactic_features * ( self.max - self.min ) + self.min
    
#     def inverse_logit_transform(self, alpha=1e-6):
#         self.galactic_features = 1 / (1 + torch.exp(-self.galactic_features))
#         self.galactic_features = (self.galactic_features - alpha) / (1 - 2 * alpha)
