
import abc
import numpy as np
import torch 
from torch import Tensor
from tqdm.auto import tqdm
from scipy.stats import qmc
from dgm.dynamics.sampler import Predictor_Corrector_SDE

class SDE(abc.ABC):
	"""SDE abstract class."""

	def __init__(self, args):
		"""Construct an SDE."""
		super().__init__()
		self.args = args
		self.dim = args.dim
		self.num_time_steps = args.num_time_steps
		self.device = args.device
		self.ODEsolver = args.ODEsolver
		self.eps = 1e-5 # t->0 lower bound for stability

	@abc.abstractmethod
	def sde(self, x: Tensor, t: Tensor):
		""" Drift and diffusion functions for the forward SDE
		"""
		pass

	def backward_sde(self, x: Tensor, context: Tensor, t: Tensor, score):
		""" Drift and diffusion functions for the backwards SDE
		"""
		xct = torch.cat([x, context, t], dim=1)
		drift, diffusion = self.sde(x, t)
		drift = drift - diffusion**2 * score(xct) 
		return drift, diffusion

	def ode(self, x: Tensor, context: Tensor, t: Tensor, score):
		""" probability-flow ODE associated with backward SDE
		"""
		xct = torch.cat([x, context, t], dim=1)
		drift, diffusion = self.sde(x, t)
		drift = drift - (1/2)*diffusion**2 * score(xct) 
		return drift

	@abc.abstractmethod
	def perturbation_kernel(self, x: Tensor, t: Tensor):
		""" the SDE perturbation kernel p_0t(x(t)|x(0)) = Normal(x(t) | mu(t) * x(0), mu(t)^2*sig^2(t))
		"""
		pass

	@abc.abstractmethod
	def alpha_fn(self, t: Tensor, M):
		""" alpha(t) for step size in Langevin dynamics for smapling. 
			M: num corrector steps
		"""
		pass

	@property
	def T(self):
		""" End time of the flow, e.g. T=1
		"""
		return 1.0

	def loss(self, target: Tensor, source: Tensor, context: Tensor, net):
		''' * Denoising Score-Matching loss *
		'''
		t = torch.rand(target.shape[:-1]+(1,), device=self.device)
		source = torch.randn_like(target) if not source.size(-1) else source 

		t =  self.eps + t * (self.T - self.eps) # t ~ Uniform(eps, T)
		_, std = self.perturbation_kernel(target, t)
		x =  target + source * std 
		xct = torch.cat([x, context, t], dim=1) 
		score = net(xct)
		loss = torch.square(score * std + source)
		return torch.mean(loss)

	@torch.no_grad()
	def sampler(self, data: Tensor, net, sampling='prob_flow_ode'):
		'''  Sample solved trejectories from source (t=0) to target (t=T)
		'''
		data = data.to(self.device)
		source = data[:, :self.dim]
		context = data[:, self.dim:]
		timesteps = torch.linspace(self.T, self.eps, self.num_time_steps, device=self.device)

		if sampling=='predictor-corrector':
			sol = Predictor_Corrector_SDE(source, context, timesteps, 
									      score=net, rsde=self.backward_sde, alpha_fn=self.alpha_fn, snr=0.16)
		if sampling=='prob_flow_ode':
			sol = NODEsolve(source, context, timesteps, self.ode, self.ODEsolver)	

		return sol

####################################################

class VariancePreservingSDE(SDE):

	def __init__(self, args, beta_min=0.1, beta_max=20.):
		super().__init__(args)
		self.beta_min = beta_min
		self.beta_max = beta_max
		self.discrete_betas = torch.linspace(self.beta_min / self.num_time_steps, 
											 self.beta_max / self.num_time_steps, 
											 self.num_time_steps)
		self.alphas = 1. - self.discrete_betas

	def sde(self, x, t):
		beta_t = self.beta_min + t * (self.beta_max - self.beta_min)
		drift = -0.5 * beta_t * x
		diffusion = torch.sqrt(beta_t)
		return drift, diffusion 

	def perturbation_kernel(self, x, t, eps=1e-5):
		coeff = -0.25 * t**2 * (self.beta_max - self.beta_min) - 0.5 * t * self.beta_min
		mean = torch.exp(coeff) * x
		std = torch.sqrt(1. - torch.exp(2. * coeff))
		return mean, std

	def alpha_fn(self, t, M):
		timestep = (t * (M - 1) / self.T).long()	
		return self.alphas.to(t.device)[timestep]

####################################################

class VarianceExplodingSDE(SDE):

	def __init__(self, args, sigma_min=0.01, sigma_max=50.):
		super().__init__(args)

		self.sigma_min = sigma_min
		self.sigma_max = sigma_max
		self.discrete_sigmas = torch.exp(torch.linspace(np.log(self.sigma_min), np.log(self.sigma_max), self.num_time_steps))

	def sde(self, x, t):
		sigma = self.sigma_min * (self.sigma_max / self.sigma_min)**t
		drift = torch.zeros_like(x)
		diffusion =  sigma * torch.sqrt(2 * torch.tensor(np.log(self.sigma_max) - np.log(self.sigma_min)))
		return drift, diffusion

	def perturbation_kernel(self, x, t):
		mean = x
		std = self.sigma_min * (self.sigma_max / self.sigma_min)**t 
		return mean, std

	def alpha_fn(self, t, M):
		return torch.ones_like(t)

####################################################

class ElucidatedSDE(SDE):

	def __init__(self, args):
		super().__init__(args)

	def sde(self, x, t):
		drift =  torch.zeros_like(x)
		diffusion = torch.sqrt(2 * t)
		return drift, diffusion 

	def perturbation_kernel(self, x, t):
		mean = x
		std = t**2
		return mean, std

	def alpha_fn(self, t, M):
		pass
