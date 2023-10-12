import torch
import nflows
from nflows.flows.base import Flow
from nflows.transforms.base import CompositeTransform
from nflows.distributions.normal import StandardNormal
from nflows.transforms.permutations import ReversePermutation
from dataclasses import dataclass
from copy import deepcopy

class DeconvolutionNormFlows:

	def __init__(self, net, configs: dataclass):
		self.device = configs.DEVICE
		self.num_mc_draws = configs.num_mc_draws
		self.num_transforms = configs.num_transforms
		self.flow_net = net
		self.permutation = ReversePermutation(features=configs.dim_input)
		self.transforms()
		self.flows = CompositeTransform(self.transforms).to(configs.DEVICE)
		self.base_distribution = StandardNormal(shape=[configs.dim_input]).to(configs.DEVICE)
		self.net = Flow(self.flows, self.base_distribution)

	def transforms(self):
		self.transforms = []
		for _ in range(self.num_transforms):
			net = deepcopy(self.flow_net)
			perm = deepcopy(self.permutation)
			self.transforms.append(net)
			self.transforms.append(perm)

	def loss(self, batch):
		""" deconvolution flow-mathcing MSE loss
		"""
		cov = batch['covariance']
		smeared = batch['smeared'] 
		cov = cov.repeat_interleave(self.num_mc_draws,0)            # ABC... -> AABBCC...
		smeared = smeared.repeat_interleave(self.num_mc_draws,0)    # ABC... -> AABBCC...
		epsilon = torch.randn_like(smeared)
		epsilon = torch.reshape(epsilon,(-1, epsilon.dim(), 1)) 
		x = smeared + torch.squeeze(torch.bmm(cov, epsilon))        # x = smeared - cov * epsilon
		x = x.to(self.device)
		logprob = torch.reshape(self.net.log_prob(x),(-1, self.num_mc_draws))
		loss = - torch.mean(torch.logsumexp(logprob, dim=-1))
		return loss + torch.log(torch.tensor(self.num_mc_draws))
