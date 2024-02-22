import torch 
from dataclasses import dataclass
from torchcfm.conditional_flow_matching import ConditionalFlowMatcher, ExactOptimalTransportConditionalFlowMatcher, SchrodingerBridgeConditionalFlowMatcher 
from torchcfm.optimal_transport import OTPlanSampler


class ConditionalFlowMatching:

	def __init__(self, config: dataclass):
		self.sigma_min = config.SIGMA
		self.coupling = config.DYNAMICS

	def define_source_target_coupling(self, batch):
		""" conditional variable z = (x_0, x1) ~ pi(x_0, x_1)
		"""		
		self.x0 = batch['source'] 
		self.x1 = batch['target']

	def sample_time(self):
		""" sample time: t ~ U[0,1]
		"""
		t = torch.rand(self.x1.shape[0], device=self.x1.device).type_as(self.x1)
		self.t = self.reshape_time(t, self.x1)

	def sample_gaussian_conditional_path(self):
		""" sample conditional path: x_t ~ p_t(x|x_0, x_1)
		"""
		mean = self.t * self.x1 + (1 - self.t) * self.x0
		std = self.sigma_min
		self.path = mean + std * torch.randn_like(mean)

	def conditional_vector_fields(self):
		""" conditional vector field (drift) u_t(x|x_0,x_1)
		"""
		self.drift = self.x1 - self.x0 

	def loss(self, model, batch):
		""" conditional flow-mathcing MSE loss
		"""
		self.define_source_target_coupling(batch)
		self.sample_time() 
		self.sample_gaussian_conditional_path()
		self.conditional_vector_fields()
		vt = model(x=self.path, t=self.t, mask=None)
		ut = self.drift.to(vt.device)
		loss = torch.square(vt - ut)
		return torch.mean(loss)

	def reshape_time(self, t, x):
		if isinstance(t, (float, int)): return t
		else: return t.reshape(-1, *([1] * (x.dim() - 1)))


class OptimalTransportCFM(ConditionalFlowMatching):
	
	def define_source_target_coupling(self, batch):
		OT = OTPlanSampler(method='exact')	
		pi = OT.get_map(batch['source'], batch['target'])		
		i,j = OT.sample_map(pi, self.x1.shape[0], replace=False)
		self.x0 = batch['target'][i]  
		self.x1 = batch['source'][j] 

class ContextOptimalTransportCFM(ConditionalFlowMatching):
	
	def define_source_target_coupling(self, batch):
		OT = OTPlanSampler(method='exact')	
		pi = OT.get_map(batch['source context'], batch['target context'])		
		i,j = OT.sample_map(pi, self.x1.shape[0], replace=False)
		self.x0 = batch['target'][i]  
		self.x1 = batch['source'][j] 


class SchrodingerBridgeCFM(ConditionalFlowMatching):
	
	def define_source_target_coupling(self, batch):
		regulator = 2 * self.sigma_min**2
		SB = OTPlanSampler(method='exact', reg=regulator)	
		pi = SB.get_map(batch['source'], batch['target'])
		i,j = SB.sample_map(pi, self.x1.shape[0], replace=False)
		self.x0 = batch['target'][i]  
		self.x1 = batch['source'][j]

	def sample_gaussian_conditional_path(self):
		mean = self.t * self.x1 + (1 - self.t) * self.x0
		std = self.sigma_min * self.t * (1 - self.t)
		self.path = mean + std * torch.randn_like(mean)

	def conditional_vector_fields(self):
		self.drift = self.x1 - self.x0 

class ContextSchrodingerBridgeCFM(ConditionalFlowMatching):
	
	def define_source_target_coupling(self, batch):
		""" conditional variable z = (x_0, x1) ~ pi(x_0, x_1)
		"""	
		regulator = 2 * self.sigma_min**2
		SB = OTPlanSampler(method='exact', reg=regulator)	
		pi = SB.get_map(batch['source context'], batch['target context'])
		i,j = SB.sample_map(pi, self.x1.shape[0], replace=False)
		self.x0 = batch['target'][i]  
		self.x1 = batch['source'][j]

	def sample_gaussian_conditional_path(self):
		""" sample conditional path: x_t ~ p_t(x|x_0, x_1)
		"""
		mean = self.t * self.x1 + (1 - self.t) * self.x0
		std = self.sigma_min * self.t * (1 - self.t)
		self.path = mean + std * torch.randn_like(mean)

	def conditional_vector_fields(self):
		""" regression objective: conditional vector field (drift) u_t(x|x_0,x_1)
		"""
		self.drift = self.x1 - self.x0 


#...TORCHCFM Wrappers

# class ConditionalFlowMatching:

# 	def __init__(self, config: dataclass):
# 		self.sigma_min = config.SIGMA
# 		self.t0 = config.T0
# 		self.t1 = config.T1

# 	def flowmatcher(self, batch):
# 		CFM = ConditionalFlowMatcher(sigma=self.sigma_min)
# 		t, xt, ut = CFM.sample_location_and_conditional_flow(batch['source'], batch['target'])
# 		self.t = (self.t1 - self.t0) * t[:, None]
# 		self.path = xt
# 		self.u = ut

# 	def loss(self, model, batch):
# 		""" conditional flow-mathcing/score-matching MSE loss
# 		"""
# 		self.flowmatcher(batch)
# 		v = model(x=self.path, t=self.t, mask=batch['mask'])
# 		u = self.u.to(v.device)
# 		loss = torch.square(v - u)
# 		return torch.mean(loss)

# class OptimalTransportFlowMatching(ConditionalFlowMatching):
	
# 	def flowmatcher(self, batch):
# 		OTFM = ExactOptimalTransportConditionalFlowMatcher(sigma=self.sigma_min)
# 		t, xt, ut = OTFM.sample_location_and_conditional_flow(batch['source'], batch['target'])
# 		self.t = (self.t1 - self.t0) * t[:, None]
# 		self.path = xt
# 		self.u = ut

# class SchrodingerBridgeFlowMatching(ConditionalFlowMatching):
	
# 	def flowmatcher(self, batch):
# 		SBFM = SchrodingerBridgeConditionalFlowMatcher(sigma=self.sigma_min, ot_method='exact')
# 		t, xt, ut = SBFM.sample_location_and_conditional_flow(batch['source'], batch['target'])
# 		self.t = (self.t1 - self.t0) * t[:, None]
# 		self.path = xt
# 		self.u = ut 