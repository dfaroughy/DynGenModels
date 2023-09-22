import torch
import numpy as np

class FormatData:

    def __init__(self, 
                 data: torch.Tensor=None,
                 cuts: dict=None):
        self.data = data
        self.cuts = cuts if cuts is not None else {'theta': None, 'phi': None, 'energy':None}

    @property 
    def theta(self): 
        return self.data[...,0]
    @property 
    def phi(self): 
        return self.data[...,1]
    @property 
    def energy(self): 
        return self.data[...,2]
   
    def selection_cuts(self, feature, cut=None):
        if cut is None: cut=[-np.inf, np.inf]
        dic={'theta':0, 'phi':1, 'energy':2}
        mask = (self.data[..., dic[feature]] >= cut[0]) & (self.data[..., dic[feature]] <= cut[1])
        self.data = self.data[mask]
        return self

    def format(self):
        self.selection_cuts(feature='energy', cut=self.cuts['energy'])
        self.selection_cuts(feature='theta', cut=self.cuts['theta'])
        self.selection_cuts(feature='phi', cut=self.cuts['phi'])
    

