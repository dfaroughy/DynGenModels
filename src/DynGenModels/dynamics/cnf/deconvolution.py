import torch
from dataclasses import dataclass

from DynGenModels.dynamics.cnf.flowmatch import SimplifiedCondFlowMatching





class DeconvolutionCondFlowMatching(SimplifiedCondFlowMatching):

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
	




class DeconvolutionFlowMatching(SimplifiedCondFlowMatching):

	def z(self, batch):
		""" conditional variable
			x0: std gauss (source)
			x1: clean data (target), x1 = x0 - cov * eps
		"""
		eps = torch.randn_like(batch['smeared'])
		self.x1 = batch['smeared'] - torch.bmm(batch['covariance'], eps.unsqueeze(-1)).squeeze(-1)
		self.x0 = torch.randn_like(self.x1)   
		self.cov = batch['covariance']

	def conditional_probability_path(self):
		""" mean and std of the Guassian conditional probability p_t(x|x_1)
		"""
		self.mean = 1
		self.std =  1 - (1 - self.sigma_min) * self.t
	

	def conditional_vector_fields(self):
		""" regression objective: conditional vector field u_t(x|x_1)
		"""
		self.u =  (1 - self.sigma_min) * torch.bmm(self.cov, self.x0.unsqueeze(-1)).squeeze(-1) 

	def sample_time(self):
		""" sample time from Uniform: t ~ U[t0, t1]
		"""
		t = (self.t1 - self.t0) * torch.rand(self.x0.shape[0], device=self.x0.device).type_as(self.x0)
		self.t = self.reshape_time(t, x=self.x0)

	def sample_path(self):
		""" sample a path: x_t ~ p_t(x|x_0)
		"""
		self.conditional_probability_path()
		self.path = self.mean * self.x1 + self.std * torch.bmm(self.cov, self.x0.unsqueeze(-1)).squeeze(-1) 

	def loss(self, batch):
		""" conditional flow-mathcing/score-matching MSE loss
		"""
		self.z(batch)
		self.conditional_vector_fields()
		self.sample_time() 
		self.sample_path()
		v = self.net(x=self.path, t=self.t)
		u = self.u 
		loss = torch.square(v - u)
		return torch.mean(loss)

	def reshape_time(self, t, x):
		""" reshape the time vector t to match dim(x).
		"""
		if isinstance(t, float): return t
		return t.reshape(-1, *([1] * (x.dim() - 1)))