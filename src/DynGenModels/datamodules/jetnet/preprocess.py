import torch

class PreprocessData:

    def __init__(self, 
                 data, 
                 stats,
                 methods: list=['center_jets', 'standardize'],
                 compute_jet_features: bool=False):
        
        self.particle_features = data[:, :-1]
        self.mask = data[:, -1].unsqueeze(-1)
        self.jet_features = self.get_jet_features() if compute_jet_features else None
        self.mean, self.std, self.min, self.max = stats 
        self.methods = methods
    
    def get_jet_features(self):
        mask = self.mask.squeeze(-1)
        eta, phi, pt = self.particle_features[:, 0], self.particle_features[:, 1], self.particle_features[:, 2]
        multiplicity = torch.sum(mask, dim=0)
        e_j  = torch.sum(mask * pt * torch.cosh(eta), dim=0)
        px_j = torch.sum(mask * pt * torch.cos(phi), dim=0)
        py_j = torch.sum(mask * pt * torch.sin(phi), dim=0)
        pz_j = torch.sum(mask * pt * torch.sinh(eta), dim=0)
        pt_j = torch.sqrt(px_j**2 + py_j**2)
        m_j  = torch.sqrt(e_j**2 - px_j**2 - py_j**2 - pz_j**2)
        eta_j = torch.asinh(pz_j / pt_j)
        phi_j = torch.atan2(py_j, px_j)
        return torch.Tensor((pt_j, eta_j, phi_j, m_j, multiplicity))

    def preprocess(self):
        for method in self.methods:
            method = getattr(self, method, None)
            if method and callable(method): method()
            else: raise ValueError('Preprocessing method {} not implemented'.format(method))
        
    def center_jets(self):
        if self.jet_features is None: raise ValueError('Jet features are not computed... set `compute_jet_features = True`')
        N, P, D = self.particle_features.shape
        jet_coords = self.jet_features[1:3] # eta, phi
        jet_coords = jet_coords.repeat(N, 1) * self.mask
        zeros = torch.zeros((N, D - 2))
        jet_coords = torch.cat((jet_coords, zeros), dim=1)
        self.particle_features -= jet_coords 

    def standardize(self,  sigma: float=1.0):
        self.particle_features = (self.particle_features * self.mean) * (sigma / self.std )
        self.particle_features *= self.mask

    def normalize(self):
        self.particle_features = (self.particle_features - self.min) / ( self.max - self.min )
        self.particle_features *= self.mask
    
    def logit_transform(self, alpha=1e-6):
        self.particle_features = self.particle_features * (1 - 2 * alpha) + alpha
        self.particle_features = torch.log(self.particle_features / (1 - self.particle_features))
        self.particle_features *= self.mask
