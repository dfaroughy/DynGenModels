import torch 
from dataclasses import dataclass


class SimplifiedCondFlowMatching:

	def __init__(self, net, config: dataclass):
		self.sigma_min = config.sigma
		self.t0 = config.t0
		self.t1 = config.t1
		self.net = net

	def z(self, batch):
		""" conditional variable
		"""
		self.x0 = batch['source'] 
		self.x1 = batch['target'] 

	def conditional_probability_path(self):
		""" mean and std of the Guassian conditional probability p_t(x|x_0,x_1)
		"""
		self.mean = (self.t - self.t0) * self.x1 + (self.t1 - self.t) * self.x0
		self.std = self.sigma_min 
	
	def conditional_vector_fields(self):
		""" regression objective: conditional vector field u_t(x|x_0,x_1)
		"""
		self.u = self.x1 - self.x0 

	def sample_time(self):
		""" sample time from Uniform: t ~ U[t0, t1]
		"""
		t = (self.t1 - self.t0) * torch.rand(self.x1.shape[0], device=self.x1.device).type_as(self.x1)
		self.t = self.reshape_time(t, x=self.x1)

	def sample_path(self):
		""" sample a path: x_t ~ p_t(x|x_0)
		"""
		self.conditional_probability_path()
		self.path = self.mean + self.std * torch.randn_like(self.x1)

	def loss(self, batch):
		""" conditional flow-mathcing/score-matching MSE loss
		"""
		self.z(batch)
		self.conditional_vector_fields()
		self.sample_time() 
		self.sample_path()
		v = self.net(x=self.path, t=self.t, mask=batch['mask'])
		u = self.u.to(v.device)
		loss = torch.square(v - u)
		return torch.mean(loss)

	def reshape_time(self, t, x):
		""" reshape the time vector t to match dim(x).
		"""
		if isinstance(t, float): return t
		return t.reshape(-1, *([1] * (x.dim() - 1)))