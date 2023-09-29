import torch
from dataclasses import dataclass

from DynGenModels.dynamics.cnf.flowmatch import SimplifiedCondFlowMatching

class DeconvolutionFlowMatching(SimplifiedCondFlowMatching):

	def z(self, batch):
		""" conditional variable
			x0: smeared data (source)
			x1: clean data (target), x1 = x0 - cov * eps
		"""
		epsilon = torch.randn_like(batch['source'])
		self.x0 = batch['source']   
		self.x1 = batch['source'] - torch.bmm(batch['covariance'], epsilon.unsqueeze(-1)).squeeze(-1)      
