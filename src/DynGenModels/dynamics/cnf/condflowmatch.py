import torch 
from dataclasses import dataclass
from torchcfm.conditional_flow_matching import ConditionalFlowMatcher, ExactOptimalTransportConditionalFlowMatcher, SchrodingerBridgeConditionalFlowMatcher 
from torchcfm.optimal_transport import OTPlanSampler

#...TORCHCFM Wrappers

class ConditionalFlowMatching:

	def __init__(self, config: dataclass):
		self.sigma_min = config.SIGMA
		self.t0 = config.T0
		self.t1 = config.T1

	def flowmatcher(self, batch):
		CFM = ConditionalFlowMatcher(sigma=self.sigma_min)
		t, xt, ut = CFM.sample_location_and_conditional_flow(batch['source'], batch['target'])
		self.t = (self.t1 - self.t0) * t[:, None]
		self.path = xt
		self.u = ut

	def loss(self, model, batch):
		""" conditional flow-mathcing/score-matching MSE loss
		"""
		self.flowmatcher(batch)
		v = model(x=self.path, t=self.t, mask=batch['mask'])
		u = self.u.to(v.device)
		loss = torch.square(v - u)
		return torch.mean(loss)

class OptimalTransportFlowMatching(ConditionalFlowMatching):
	
	def flowmatcher(self, batch):
		OTFM = ExactOptimalTransportConditionalFlowMatcher(sigma=self.sigma_min)
		t, xt, ut = OTFM.sample_location_and_conditional_flow(batch['source'], batch['target'])
		self.t = (self.t1 - self.t0) * t[:, None]
		self.path = xt
		self.u = ut

class SchrodingerBridgeFlowMatching(ConditionalFlowMatching):
	
	def flowmatcher(self, batch):
		SBFM = SchrodingerBridgeConditionalFlowMatcher(sigma=self.sigma_min, ot_method='exact')
		t, xt, ut = SBFM.sample_location_and_conditional_flow(batch['source'], batch['target'])
		self.t = (self.t1 - self.t0) * t[:, None]
		self.path = xt
		self.u = ut 

#... MY code

class ConditionalFlowMatching:

	def __init__(self, config: dataclass, coupling: str = None):
		self.sigma_min = config.SIGMA
		self.batch_size = config.BATCH_SIZE
		self.coupling = config.DYNAMICS

	def source_target_coupling(self, batch):
		""" conditional variable z = (x_0, x1) ~ pi(x_0, x_1)
		"""	
		if self.coupling == 'OptimalControlFlowMatching':
			OT = OTPlanSampler(method='exact')	
			pi = OT.get_map(batch['source'], batch['target'])		
			self.i, self.j = OT.sample_map(pi, self.batch_size, replace=False)
			self.x0 = batch['target'][self.i]  
			self.x1 = batch['source'][self.j] 

		elif self.coupling == 'SchrodingerBridgeFlowMatching':
			OT = OTPlanSampler(method='exact', reg=2 * self.sigma_min**2)	
			pi = OT.get_map(batch['source'], batch['target'])		
			self.i, self.j = OT.sample_map(pi, self.batch_size, replace=False)
			self.x0 = batch['target'][self.i]  
			self.x1 = batch['source'][self.j] 
		
		elif self.coupling == 'ContextOptimalControlFlowMatching':
			OT = OTPlanSampler(method='exact')	
			pi = OT.get_map(batch['source context'], batch['target context'])		
			self.i, self.j = OT.sample_map(pi, self.batch_size, replace=False)
			self.x0 = batch['target'][self.i]  
			self.x1 = batch['source'][self.j] 

		elif self.coupling == 'ContextSchrodingerBridgeFlowMatching':
			OT = OTPlanSampler(method='exact', reg=2 * self.sigma**2)	
			pi = OT.get_map(batch['source'], batch['target'] )		
			self.i, self.j = OT.sample_map(pi, self.batch_size, replace=False)
			self.x0 = batch['target'][self.i]  
			self.x1 = batch['source'][self.j] 

		else:	
			self.x0 = batch['source'] 
			self.x1 = batch['target']

	def conditional_probability_path(self):
		""" mean and std of the Guassian conditional probability p_t(x|x_0,x_1)
		"""
		t = self.reshape_time(self.t, x=self.x1)
		self.mean = t * self.x1 + (1 - t) * self.x0
		self.std = self.sigma_min * (torch.sqrt(t * (1 - t)) if self.coupling == 'SB' else 1.0 )
	
	def conditional_vector_fields(self):
		""" regression objective: conditional vector field u_t(x|x_0,x_1)
		"""
		self.u = self.x1 - self.x0 

	def sample_time(self):
		""" sample time from Uniform: t ~ U[0,1]
		"""
		torch.manual_seed(12345)
		self.t = torch.rand(self.batch_size, device=self.x1.device).type_as(self.x1)
	

	def sample_conditional_path(self):
		""" sample a path: x_t ~ p_t(x|x_0, x_1)
		"""
		torch.manual_seed(12345)
		self.conditional_probability_path()
		self.path = self.mean + self.std * torch.randn_like(self.x1)

	def loss(self, model, batch):
		""" conditional flow-mathcing MSE loss
		"""
		self.source_target_coupling(batch)
		self.conditional_vector_fields()
		self.sample_time() 
		self.sample_conditional_path()
		v = model(x=self.path, t=self.t.unsqueeze(-1), mask=None)
		u = self.u.to(v.device)
		loss = torch.square(v - u)
		return torch.mean(loss)

	def reshape_time(self, t, x):
		if isinstance(t, (float, int)):
			return t
		return t.reshape(-1, *([1] * (x.dim() - 1)))
