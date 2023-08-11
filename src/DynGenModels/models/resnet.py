import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

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


class ResNet(nn.Module):

	def __init__(self, 
	            dim=3, 
	            dim_context=0, 
		        dim_hidden=64, 
		        num_layers=5, 
		        device='cpu'):
		
		super(ResNet, self).__init__()

		self.position_embedding = nn.Linear(dim + dim_context, dim_hidden//2).to(device)
		self.time_embedding = nn.Sequential(FourierFeatures(dim=dim_hidden//2),
										  nn.Linear(dim_hidden//2, dim_hidden//2)
										  ).to(device)

		block = ResBlock(dim=dim_hidden, device=device)
		self.blocks = nn.ModuleList([block for _ in range(num_layers)]).to(device)                         
		self.final_layer = nn.Linear(dim_hidden, dim).to(device)

	def forward(self, t, x, context=None, mask=None): 
		mask = mask.unsqueeze(-1) if mask is not None else torch.ones_like(x[..., 0]).unsqueeze(-1)
		xc = torch.cat([x, context], dim=-1) if context is not None else x    							# (B, P, dim + dim_context)
		xct = torch.cat([self.position_embedding(xc) * mask, self.time_embedding(t) * mask], dim=-1)    # (B, P, dim_hidden)
		for block in self.blocks: xct = block(xct)
		f = self.final_layer(xct)
		return f