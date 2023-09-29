import torch
from dataclasses import dataclass

from DynGenModels.dynamics.nf.normflows import NormalizingFlow

class DeconvolutionNormFlows( NormalizingFlow):
		    
	def __init__(self, net, configs: dataclass):
		super().__init__(net, configs)
		self.num_mc_draws = configs.num_mc_draws

	def loss(self, batch):
		""" deconvolution flow-mathcing MSE loss
		"""
		cov = batch['covariance']
		source = batch['source'] 
		cov = cov.repeat_interleave(self.num_mc_draws,0)            # ABC... -> AABBCC...
		source = source.repeat_interleave(self.num_mc_draws,0)      # ABC... -> AABBCC...

		epsilon = torch.randn_like(source)
		target = torch.bmm(cov, epsilon.unsqueeze(-1)).squeeze(-1)
		target = source + torch.bmm(cov, epsilon.unsqueeze(-1)).squeeze(-1)  # data + sigma * eps

		loss = - torch.mean(torch.logsumexp(torch.reshape(self.flow.log_prob(target),(-1, self.num_mc_draws)), dim=-1))

		return torch.mean(loss)
