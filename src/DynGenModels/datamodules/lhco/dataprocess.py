from typing import Any
import torch
import numpy as np

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
    