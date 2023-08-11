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


class DeepSetNet(nn.Module):
	def __init__(self, dim=3, dim_context=0, dim_hidden=64, device='cpu'):
		super(DeepSetNet, self).__init__()

		self.position_embedding = nn.Linear(dim + dim_context, dim_hidden//2).to(device)
		self.time_embedding = nn.Sequential(FourierFeatures(dim=dim_hidden//2),
										  nn.Linear(dim_hidden//2, dim_hidden//2)
										  ).to(device)

		self.phi = nn.Sequential(
								nn.Linear(dim_hidden, dim_hidden),
								nn.ReLU(),
								nn.Linear(dim_hidden, dim_hidden),
								nn.ReLU(),
								nn.Linear(dim_hidden, dim_hidden),
								nn.ReLU(),
								nn.Linear(dim_hidden, dim_hidden),
								nn.ReLU(),
								nn.Linear(dim_hidden, dim_hidden),
								nn.ReLU()
								 ).to(device)
		
		self.rho = nn.Sequential(
								nn.Linear(dim_hidden, dim_hidden),
								nn.ReLU(),
								nn.Linear(dim_hidden, dim_hidden),
								nn.ReLU(),
								nn.Linear(dim_hidden, dim_hidden),
								nn.ReLU(),
								nn.Linear(dim_hidden, dim_hidden),
								nn.ReLU(),
								nn.Linear(dim_hidden, dim),
								nn.ReLU()
								).to(device)

	def forward(self, t, x, context=None, mask=None): 
		xc = torch.cat([x, context], dim=-1) if context is not None else x
		xct = torch.cat([self.position_embedding(xc), self.time_embedding(t)], dim=-1) 
		h = self.phi(xct)                                             # shape: (N, m, hidden_dim)
		h = torch.sum(h + xct, dim=0) + torch.mean(h + xct, dim=0)    # sum pooling shape: (N, hidden_dim)
		h = self.rho(h)                                               # shape: (N, output_dim)
		return h


class DeepSets(nn.Module):
    ''' Wrapper class for the Deep Sets architecture'''
    def __init__(self, model_config):
        super(DeepSets, self).__init__()

        self.dim_features = model_config.dim_input
        self.device = model_config.device

        self.deepset = DeepSetNet(dim=model_config.dim_input, 
                                num_classes=model_config.dim_output,
                                dim_hidden=model_config.dim_hidden, 
                                num_layers_1=model_config.num_layers_1,
                                num_layers_2=model_config.num_layers_2,
                                device=model_config.device)
        
        self.criterion = nn.CrossEntropyLoss()
        
    def forward(self, features, mask):
        return self.deepset.forward(features, mask)
    
    def loss(self, batch):
        features = batch['particle_features'].to(self.device)
        labels = batch['label'].to(self.device)
        mask = batch['mask'].to(self.device)
        output = self.forward(features, mask)
        loss =  self.criterion(output, labels)
        return loss

    @torch.no_grad()
    def predict(self, batch): 
        features = batch['particle_features'].to(self.device)
        mask = batch['mask'].to(self.device)
        logits = self.forward(features, mask)
        probs = F.softmax(logits, dim=1)
        return probs.detach().cpu()  