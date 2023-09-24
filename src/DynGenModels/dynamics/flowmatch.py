import torch 

class SimplifiedCondFlowMatching:

	def __init__(self, net, sigma_min=1e-6):
		self.sigma_min = sigma_min
		self.net = net

	def z(self, x0, x1):
		""" conditional variable
		"""
		self.x0 = x0 # source
		self.x1 = x1 # target

	def conditional_vector_field(self, x, t):
		""" regressed vector field u_t(x|x_0,x_1)
		"""
		u = self.x1 - self.x0
		return u 

	def gaussian_probability_path(self, t):
		""" mean and std of the Guassian conditional probability p_t(x|x_0,x_1)
		"""
		mean  = t * self.x1 + (1-t) * self.x0
		std = self.sigma_min
		return mean, std 
	
	@property
	def T(self):
		""" end time of the flow
		"""
		return 1.0
	
	def loss(self, batch):
		""" conditional flow-mathcing MSE loss
		"""

		target = batch['target']
		source = batch['source'] 
		self.z(x0=source, x1=target)

		t = torch.rand(target.shape[0], device=target.device).type_as(target)
		t = self.reshape_time(t, x=target)

		mu_t, sigma_t = self.gaussian_probability_path(t)
		noise = torch.randn_like(source)
		x_t = mu_t + sigma_t * noise
		u_t = self.conditional_vector_field(x_t, t)
		v_t = self.net(x=x_t, t=t)
		
		loss = torch.square(v_t - u_t)

		return torch.mean(loss)
	
	def reshape_time(self, t, x):
		""" reshape the time vector t to dim(x).
		"""
		if isinstance(t, float):
			return t
		return t.reshape(-1, *([1] * (x.dim() - 1)))
