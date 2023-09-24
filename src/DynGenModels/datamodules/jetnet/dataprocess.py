import torch

class FormatData:

    def __init__(self, 
                 data: torch.Tensor=None,
                 max_num_jets: int=None,
                 max_num_constituents: int=150,
                 particle_features: list=['eta_rel', 'phi_rel', 'pt_rel'],
                 remove_negative_pt: bool=False
                ):
        
        self.data = data
        self.max_num_jets = data.shape[0] if max_num_jets is None else max_num_jets
        self.max_num_consts = max_num_constituents
        self.particle_features = particle_features
        self.remove_negative_pt = remove_negative_pt

    def data_rank(self, n):
        return len(self.data.shape) == n

    def format(self):
        if self.data_rank(3):  
            self.zero_padding()
            if self.remove_negative_pt: self.remove_neg_pt() 
            self.get_particle_features()
            self.trim_dataset()
        if self.data_rank(2): 
            # TODO
            pass
    
    def zero_padding(self):
        N, P, D = self.data.shape
        if P < self.max_num_consts:
            zero_rows = torch.zeros(N, self.max_num_consts - P, D)
            self.data = torch.cat((self.data, zero_rows), dim=1)
        else: pass 

    def get_particle_features(self, masked: bool=True): 
        pf = {}
        pf['eta_rel'] = self.data[..., 0, None]
        pf['phi_rel'] = self.data[..., 1, None]
        pf['pt_rel'] = self.data[..., 2, None]
        pf['e_rel'] = pf['pt_rel'] * torch.cosh(pf['eta_rel'])
        pf['log_pt_rel'] = torch.log(pf['pt_rel'])
        pf['log_e_rel'] = torch.log(pf['e_rel'])
        pf['R'] = torch.sqrt(pf['eta_rel']**2 + pf['phi_rel']**2)
        features = [pf[f] for f in self.particle_features]
        if masked:
            mask = (self.data[..., 2] != 0).int().unsqueeze(-1) 
            features += [mask]
        self.data = torch.cat(features, dim=-1)
        self.pt_order()        

    def remove_neg_pt(self):
        data_clip = torch.clone(self.data)    
        self.data = torch.zeros_like(self.data)
        self.data[data_clip[..., 2] >= 0.0] = data_clip[data_clip[..., 2] >= 0.0]
        self.pt_order()
    
    def trim_dataset(self):
        if self.data_rank(3): 
            self.data = self.data[:self.max_num_jets, :self.max_num_consts, :]
        if self.data_rank(2): 
            self.data = self.data[:self.max_num_jets, :] 

    def pt_order(self):
        idx = None
        for i, f in enumerate(self.particle_features):
            if 'pt' in f: 
                idx = i
                break
        if idx is None:
            raise ValueError('No pt feature found in particle features list for pt_order() method')
        if self.data_rank(3): 
            _ , i = torch.sort(torch.abs(self.data[:, :, idx]), dim=1, descending=True) 
            self.data = torch.gather(self.data, 1, i.unsqueeze(-1).expand_as(self.data)) 
        if self.data_rank(2):  
            _ , i = torch.sort(torch.abs(self.data[:, idx]), dim=1, descending=True) 
            self.data = torch.gather(self.data, 1, i.unsqueeze(-1).expand_as(self.data)) 


class PreprocessData:

    def __init__(self, 
                 data, 
                 stats,
                 methods: list=['standardize']
                 ):
        
        self.particle_features = data[:, :-1]
        self.mask = data[:, -1].unsqueeze(-1)
        self.jet_features = self.get_jet_features() if 'compute_jet_features' in methods else None
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
            if method == 'compute_jet_features': continue
            method = getattr(self, method, None)
            if method and callable(method): method()
            else: raise ValueError('Preprocessing method {} not implemented'.format(method))
        
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
