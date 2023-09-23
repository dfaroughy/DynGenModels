import torch

class PreProcessData:

    def __init__(self, 
                 data, 
                 stats,
                 methods: list=['standardize']
                 ):
        
        self.galactic_features = data
        self.mean, self.std, self.min, self.max = stats 
        self.methods = methods

    def preprocess(self):
        for method in self.methods:
            method = getattr(self, method, None)
            if method and callable(method): method()
            else: raise ValueError('Preprocessing method {} not implemented'.format(method))
        
    def standardize(self,  sigma: float=1.0):
        self.galactic_features = (self.galactic_features - self.mean) * (sigma / self.std )

    def normalize(self):
        self.galactic_features = (self.galactic_features - self.min) / ( self.max - self.min )
    
    def logit_transform(self, alpha=1e-6):
        self.galactic_features = self.galactic_features * (1 - 2 * alpha) + alpha
        self.galactic_features = torch.log(self.galactic_features / (1 - self.galactic_features))

class PostProcessData:

    def __init__(self, 
                 data, 
                 stats,
                 methods: list=['inverse_standardize']
                 ):
        
        self.galactic_features = data
        self.mean, self.std, self.min, self.max = stats 
        self.methods = methods

    def postprocess(self):
        for method in self.methods:
            method = getattr(self, method, None)
            if method and callable(method): method()
            else: raise ValueError('Preprocessing method {} not implemented'.format(method))
        
    def inverse_standardize(self,  sigma: float=1.0):
        self.galactic_features = self.galactic_features * (self.std / sigma) + self.mean

    def inverse_normalize(self):
        self.galactic_features = self.galactic_features * ( self.max - self.min ) + self.min
    
    def inverse_logit_transform(self, alpha=1e-6):
        self.galactic_features = 1 / (1 + torch.exp(-self.galactic_features))
        self.galactic_features = (self.galactic_features - alpha) / (1 - 2 * alpha)
