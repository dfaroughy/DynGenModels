import abc
import torch 

class ConditionalFlowMatching(abc.ABC):
	"""Conditional Flow-Matching abstract class.
	"""
	def __init__(self):
		super().__init__()


	@abc.abstractmethod
	def z(self, x0: torch.Tensor, x1: torch.Tensor, context: torch.Tensor):
		""" x0: source data
			x1: target data
		"""
		pass

	@abc.abstractmethod
	def cond_vector_field(self, x: torch.Tensor, t: torch.Tensor):
		""" u_t(x,z) vector field
		"""
		pass

	@abc.abstractmethod
	def probability_path(self, t: torch.Tensor):
		"""p_t(x,z) perturbation kernel 
		"""
		pass

	@property
	def T(self):
		"""End time of the flow, e.g. T=1
		"""
		t_1 = 1.0
		return t_1

	def loss(self, 
	         target: torch.Tensor,
	         source: torch.Tensor, 
	         context: torch.Tensor,
             mask: torch.Tensor,
	         model: torch.nn.Module):
		
		'''  Flow-Matching MSE loss
		'''
		
		t = torch.rand(target.shape[:-1], device=target.device).unsqueeze(-1)
		self.z(x0=source, x1=target, context=context)
		mean, std = self.probability_path(t)
		x = mean + std * torch.randn_like(source)
		v = model(t=t, x=x, context=context, mask=mask)
		u = self.cond_vector_field(x, t)
		loss = torch.square(v - u)
		return torch.mean(loss)


####################################################

class SimpleCFM(ConditionalFlowMatching):

	def __init__(self, sigma_min=1e-6):
		super().__init__()
		self.sigma_min = sigma_min

	def z(self, x0, x1, context):
		self.x0 = x0
		self.x1 = x1 

	def cond_vector_field(self, x, t):
		u = self.x1 - self.x0
		return u

	def probability_path(self, t):
		mean  = t * self.x1 + (1-t) * self.x0
		std = self.sigma_min
		return mean, std




