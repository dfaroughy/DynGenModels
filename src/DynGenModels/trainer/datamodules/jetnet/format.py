import torch

class FormatData:

    def __init__(self, 
                 data: torch.Tensor=None,
                 num_jets: int=None,
                 num_constituents: int=150,
                 particle_features: list=['eta_rel', 'phi_rel', 'pt_rel'],
                 remove_negative_pt: bool=False
                ):
        
        self.data = data
        self.num_jets = data.shape[0] if num_jets is None else num_jets
        self.num_consts = num_constituents
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
        if P < self.num_consts:
            zero_rows = torch.zeros(N, self.num_consts - P, D)
            self.data =  torch.cat((self.data, zero_rows), dim=1)
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
            self.data = self.data[:self.num_jets, :self.num_consts, :]
        if self.data_rank(2): 
            self.data = self.data[:self.num_jets, :] 

    def pt_order(self):
        for i, f in enumerate(self.particle_features):
            if 'pt' in f: 
                idx = i
                break
        if self.data_rank(3): 
            _ , i = torch.sort(torch.abs(self.data[:, :, idx]), dim=1, descending=True) 
            self.data = torch.gather(self.data, 1, i.unsqueeze(-1).expand_as(self.data)) 
        if self.data_rank(2):  
            _ , i = torch.sort(torch.abs(self.data[:, idx]), dim=1, descending=True) 
            self.data = torch.gather(self.data, 1, i.unsqueeze(-1).expand_as(self.data)) 
