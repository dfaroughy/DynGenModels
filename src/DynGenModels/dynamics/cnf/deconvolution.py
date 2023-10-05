import torch
from dataclasses import dataclass

from DynGenModels.dynamics.cnf.flowmatch import SimplifiedCondFlowMatching

class DeconvolutionFlowMatching(SimplifiedCondFlowMatching):

	def z(self, batch):
		""" conditional variable
			x0: smeared data (source)
			x1: clean data (target), x1 = x0 - cov * eps
		"""
		epsilon = torch.randn_like(batch['smeared'])
		self.x1 = batch['smeared']
		self.x0 = batch['smeared'] - torch.bmm(batch['covariance'], epsilon.unsqueeze(-1)).squeeze(-1)      

	def loss(self, batch):
		""" conditional flow-mathcing/score-matching MSE loss
		"""
		self.z(batch)
		self.conditional_vector_fields()
		self.sample_time() 
		self.sample_path()
		v = self.net(x=self.path, t=self.t)
		u = self.u 
		loss = torch.square(v - self.t*u)
		return torch.mean(loss)