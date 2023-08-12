import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

#...architecture classes

class FourierFeatures(nn.Module):

    def __init__(self, dim, scale=30.0):
        super().__init__()
        half_dim = dim // 2
        self.W = nn.Parameter(torch.randn(half_dim) * scale, requires_grad=False)

    def forward(self, z):
        z_proj = z * self.W.to(z.device) * 2 * np.pi
        return torch.cat([torch.sin(z_proj), torch.cos(z_proj)], dim=-1)


class DeepSet(nn.Module):
	def __init__(self, 
              	dim=3, 
                dim_context=0,
                dim_hidden=64, 
                num_layers_1=2,
                num_layers_2=2,
                pooling='mean_sum',
                device='cpu'):
                
		super(DeepSet, self).__init__()
                
		dim_hidden_pool = (2 if pooling=='mean_sum' else 1) * dim_hidden
		self.x_embedding = nn.Linear(dim + dim_context, dim_hidden//2).to(device)
		self.t_embedding = nn.Sequential(FourierFeatures(dim=dim_hidden//2), nn.Linear(dim_hidden//2, dim_hidden//2)).to(device)
		layers_1 = [nn.Linear(dim_hidden, dim_hidden), nn.LeakyReLU()] * num_layers_1 
		layers_2 = [nn.Linear(dim_hidden_pool, dim_hidden), nn.LeakyReLU()] + [nn.Linear(dim_hidden, dim_hidden), nn.LeakyReLU()] * (num_layers_2 - 2) + [nn.Linear(dim_hidden, dim), nn.LeakyReLU()]
		self.phi = nn.Sequential(*layers_1).to(device)
		self.rho = nn.Sequential(*layers_2).to(device)
		self.pool = pooling
                
	def forward(self, t, x, context=None, mask=None): 
		mask = mask.unsqueeze(-1) if mask is not None else torch.ones_like(x[..., 0]).unsqueeze(-1)
		xc = torch.cat([x, context], dim=-1) if context is not None else x
		xct = torch.cat([self.x_embedding(xc), self.t_embedding(t)], dim=-1) 
		h = self.phi(xct)
		h = h + xct
		h_sum = (h * mask).sum(1, keepdim=False)
		h_mean = h_sum / mask.sum(1, keepdim=False)         
		if self.pool == 'sum': f = h_sum
		elif self.pool == 'mean': f = h_mean
		elif self.pool == 'mean_sum': f = torch.cat([h_mean, h_sum], dim=-1)
		return self.rho(f)


class DeepSetUNet(nn.Module):
    pass # TODO

class DeepSets(nn.Module):
    ''' Wrapper class for the Deep Sets architecture'''
    def __init__(self, model_config):
        super(DeepSets, self).__init__()
        self.device = model_config.device
        self.deepset = DeepSet(dim=model_config.dim_input, 
                                  dim_hidden=model_config.dim_hidden, 
                                  dim_context=model_config.dim_context,	
                                  num_layers_1=model_config.num_layers_1,
                                  num_layers_2=model_config.num_layers_2,
                                  pooling=model_config.pooling,
                                  device=model_config.device)
                
    def forward(self, t, x, context, mask):
        return self.deepset.forward(t, x, context, mask)
    
