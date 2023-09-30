import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

class _MLP(torch.nn.Module):
    def __init__(self, dim, out_dim=None, w=64, time_varying=False, device='cpu'):
        super().__init__()

        self.time_varying = time_varying
        if out_dim is None: out_dim = dim

        self.net = torch.nn.Sequential(
            torch.nn.Linear(dim + (1 if time_varying else 0), w),
            torch.nn.SELU(),
            torch.nn.Linear(w, w),
            torch.nn.SELU(),
            torch.nn.Linear(w, w),
            torch.nn.SELU(),
            torch.nn.Linear(w, out_dim)).to(device)

    def forward(self, x):
        return self.net(x)

class MLP(nn.Module):
    ''' Wrapper class for the MLP architecture
    '''
    def __init__(self, config):
        super(MLP, self).__init__()

        self.mlp = _MLP(dim=config.dim_input, 
                       out_dim=None,
                       w=config.dim_hidden, 
                       time_varying=True,
                       device=config.device)
                
    def forward(self, t, x, context=None, mask=None):
        x = torch.cat([x, t], dim=-1)
        return self.mlp.forward(x)
    
class FourierFeatures(nn.Module):

    def __init__(self, dim, scale=30.0):
        super().__init__()
        half_dim = dim // 2
        self.W = nn.Parameter(torch.randn(half_dim) * scale, requires_grad=False)

    def forward(self, z):
        z_proj = z * self.W.to(z.device) * 2 * np.pi
        return torch.cat([torch.sin(z_proj), torch.cos(z_proj)], dim=-1)

class ResBlock(nn.Module):
	def __init__(self, dim, device, dropout=0.0, preactivation=True):
		super(ResBlock, self).__init__()

		self.preactivation = preactivation
		self.activ = nn.ModuleList([nn.BatchNorm1d(dim, eps=1e-3), nn.ReLU()]).to(device)
		self.block = nn.ModuleList([nn.Linear(dim, dim),
									nn.BatchNorm1d(dim, eps=1e-3),
									nn.ReLU(),
									nn.Dropout(p=dropout), 
									nn.Linear(dim, dim)]).to(device)
	def forward(self, x):
		if self.preactivation:
			h = self.activ[0](x)
			h = self.activ[1](h)
		else:
			h = x
		for i in range(len(self.block)): h = self.block[i](h)
		f = h + x # skip connection
		if not self.preactivation:
			f = self.activ[0](f)
			f = self.activ[1](f)
		return f


class _ResNet(nn.Module):

	def __init__(self, 
	            dim=3, 
	            dim_context=0, 
		        dim_hidden=64, 
		        num_layers=5, 
		        device='cpu'):
		
		super(_ResNet, self).__init__()

		self.position_embedding = nn.Linear(dim + dim_context, dim_hidden//2).to(device)
		self.time_embedding = nn.Sequential(FourierFeatures(dim=dim_hidden//2), nn.Linear(dim_hidden//2, dim_hidden//2) ).to(device)
		block = ResBlock(dim=dim_hidden, device=device)
		self.blocks = nn.ModuleList([block for _ in range(num_layers)]).to(device)                         
		self.final_layer = nn.Linear(dim_hidden, dim).to(device)

	def forward(self, t, x, context=None, mask=None): 
		t_emb = self.time_embedding(t) 
		x_emb = self.position_embedding(x) 
		xt = torch.cat([x_emb, t_emb], dim=-1)    # (B, P, dim_hidden)
		for block in self.blocks: xt = block(xt)
		f = self.final_layer(xt)
		return f
	
class ResNet(nn.Module):
    ''' Wrapper class for the ResNet architecture'''
    def __init__(self, model_config):
        super(ResNet, self).__init__()
        self.resnet = _ResNet(dim=model_config.dim_input, 
                               dim_hidden=model_config.dim_hidden, 
                               num_layers=model_config.num_layers,
                               device=model_config.device)
                
    def forward(self, t, x, context=None, mask=None):
        return self.resnet.forward(t, x, context, mask)

	