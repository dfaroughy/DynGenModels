import abc
import torch 

# class ConditionalFlowMatching(abc.ABC):

# 	def __init__(self, model):
# 		super().__init__()
# 		self.model = model

# 	@abc.abstractmethod
# 	def z(self, x0: torch.Tensor, x1: torch.Tensor, context: torch.Tensor=None):
# 		""" source-target coupling
# 			x0: source data, x1: target data
# 		"""
# 		pass

# 	@abc.abstractmethod
# 	def cond_vector_field(self, x: torch.Tensor, t: torch.Tensor):
# 		""" u_t(x,z) conditional vector field
# 		"""
# 		pass

# 	@abc.abstractmethod
# 	def probability_path(self, t: torch.Tensor):
# 		"""p_t(x,z) conditional perturbation kernel 
# 		"""
# 		pass

# 	@property
# 	def T(self):
# 		"""End time of the flow, e.g. T=1
# 		"""
# 		return 1.0

# 	def loss(self, 
# 			 target: torch.Tensor,
# 	         source: torch.Tensor=None, 
# 	         context: torch.Tensor=None,
#              mask: torch.Tensor=None
# 	         ):
		
# 		'''  Flow-Matching MSE loss
# 		'''
		
# 		t = torch.rand(target.shape[:-1], device=target.device).unsqueeze(-1)
		
# 		self.z(x0=source, x1=target, context=context)
# 		mean, std = self.probability_path(t)
# 		x = mean + std * torch.randn_like(source)
# 		v = self.model(t=t, x=x, context=context, mask=mask)
# 		u = self.cond_vector_field(x, t) * mask
# 		loss = torch.square(v - u)
# 		return torch.mean(loss)

####################################################

class SimplifiedCFM:

	def __init__(self, model, sigma_min=1e-6):
		self.sigma_min = sigma_min
		self.model = model

	def z(self, x0, x1, context=None):
		self.x0 = x0
		self.x1 = x1 

	def cond_vector_field(self, x, t):
		u = self.x1 - self.x0
		return u 

	def probability_path(self, t):
		mean  = t * self.x1 + (1-t) * self.x0
		std = self.sigma_min
		return mean, std 
	
	@property
	def T(self):
		"""End time of the flow
		"""
		return 1.0
	
	def loss(self, batch):

		target = batch['target']
		source = batch['source'] 
		context = None # batch['context']
		mask = batch['mask']

		t = torch.rand(target.shape[:-1], device=target.device).unsqueeze(-1)
		
		self.z(x0=source, x1=target, context=context)
		mean, std = self.probability_path(t)
		x = mean + std * torch.randn_like(source)
		v = self.model(t=t, x=x, context=context, mask=mask)
		print(v)
		u = self.cond_vector_field(x, t)
		u *= mask.unsqueeze(-1)

		print(v.shape, u.shape)
		loss = torch.square(v - u)
		return torch.mean(loss)
