import torch
from dataclasses import dataclass

from DynGenModels.dynamics.nf.normflows import NormalizingFlow

class DeconvolutionNormFlows(NormalizingFlow):
		    
	def __init__(self, net, configs: dataclass):
		super().__init__(net, configs)
		self.num_mc_draws = configs.num_mc_draws

	def loss(self, batch):
		""" deconvolution flow-mathcing MSE loss
		"""
		cov = batch['covariance']
		smeared = batch['smeared'] 
		# stats = batch['summary_stats']
		cov = cov.repeat_interleave(self.num_mc_draws,0)            # ABC... -> AABBCC...
		smeared = smeared.repeat_interleave(self.num_mc_draws,0)      # ABC... -> AABBCC...
		epsilon = torch.randn_like(smeared)
		target = smeared + torch.bmm(cov, epsilon.unsqueeze(-1)).squeeze(-1)  # data + sigma * eps
		loss = - torch.mean(torch.logsumexp(torch.reshape(self.flow.log_prob(target),(-1, self.num_mc_draws)), dim=-1))
		return torch.mean(loss)


def transform(x, mu, sigma):
	x = torch.log(1-x/x)
	return (x - mu) / sigma
