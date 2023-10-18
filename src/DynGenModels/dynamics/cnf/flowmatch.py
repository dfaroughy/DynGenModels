import torch 
from dataclasses import dataclass


class FlowMatching:

	def __init__(self, config: dataclass):
		self.sigma_min = config.sigma
		self.t0 = config.t0
		self.t1 = config.t1

	def z(self, batch):
		""" conditional variable
		"""
		self.x1 = batch['target'] 
		self.x0 = torch.randn_like(self.x1) 

	def conditional_probability_path(self):
		""" mean and std of the Guassian conditional probability p_t(x|x_1)
		"""
		self.mean = self.t
		self.std =  1 - (1 - self.sigma_min) * self.t
	
	def conditional_vector_fields(self):
		""" regression objective: conditional vector field u_t(x|x_1)
		"""
		self.u =  self.x1 - (1 - self.sigma_min) * self.x0 

	def sample_time(self):
		""" sample time from Uniform: t ~ U[t0, t1]
		"""
		t = (self.t1 - self.t0) * torch.rand(self.x0.shape[0], device=self.x0.device).type_as(self.x0)
		self.t = self.reshape_time(t, x=self.x0)

	def sample_path(self):
		""" sample a path: x_t ~ p_t(x|x_0)
		"""
		self.conditional_probability_path()
		self.path = self.mean * self.x1 + self.std * self.x0

	def loss(self, model, batch):
		""" conditional flow-mathcing/score-matching MSE loss
		"""
		self.z(batch)
		self.conditional_vector_fields()
		self.sample_time() 
		self.sample_path()
		v = model(x=self.path, t=self.t)
		u = self.u.to(v.device)
		loss = torch.square(v - u)
		return torch.mean(loss)

	def reshape_time(self, t, x):
		""" reshape the time vector t to match dim(x).
		"""
		if isinstance(t, float): return t
		return t.reshape(-1, *([1] * (x.dim() - 1)))
