import torch
import torch.nn as nn
from dataclasses import dataclass
import copy

from nflows.flows.base import Flow
from nflows.transforms.base import CompositeTransform
from nflows.distributions.normal import StandardNormal
from nflows.transforms.permutations import Permutation
from nflows.transforms.autoregressive import MaskedPiecewiseRationalQuadraticAutoregressiveTransform
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform
from nflows.transforms.coupling import PiecewiseRationalQuadraticCouplingTransform
from nflows.nn.nets.resnet import ResidualNet

#...MAFs:


class MAFPiecewiseRQS(nn.Module):
	''' Wrapper class for the MAF with rational quadratic splines architecture
	'''
	def __init__(self, configs: dataclass):
		super(MAFPiecewiseRQS, self).__init__()
		self.dim = configs.dim_input
		self.device = configs.DEVICE
		self.num_transforms = configs.num_transforms
		self.flow_net  = _MAFPiecewiseRQS(configs)
		self.get_permutation(configs.permutation)
		self.transforms()
		self.flows = CompositeTransform(self.transforms).to(configs.DEVICE)
		self.base_distribution = StandardNormal(shape=[configs.dim_input]).to(configs.DEVICE)
		self.net = Flow(self.flows, self.base_distribution)

	def get_permutation(self, perm):
		k = list(range(self.dim))
		if 'cycle' in perm:
			N = int(perm.split('-')[0]) 
			assert N < self.dim 
			self.permutation = Permutation(torch.tensor(k[-N:] + k[:-N]))
		elif 'reverse' in perm:
			self.permutation = Permutation(torch.tensor(k[::-1]))

	def transforms(self):
		self.transforms = []
		for _ in range(self.num_transforms):
			flow = copy.deepcopy(self.flow_net)
			perm = copy.deepcopy(self.permutation)
			self.transforms.append(flow)
			self.transforms.append(perm)

	def log_prob(self, x):
		return self.net.log_prob(x)
	
	def sample(self, num_samples):
		return self.net.sample(num_samples)

class _MAFPiecewiseRQS(nn.Module):
	''' Wrapper class for the MAF with rational quadratic splines architecture
	'''
	def __init__(self, configs):
		super(_MAFPiecewiseRQS, self).__init__() 
		self.device = configs.DEVICE
		self.maf = MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
						features=configs.dim_input,
						hidden_features=configs.dim_hidden,
						num_bins=configs.num_bins,
						tails=configs.tails,
						tail_bound=configs.tail_bound,
						num_blocks=configs.num_blocks,
						use_residual_blocks=configs.use_residual_blocks,
						dropout_probability=configs.dropout,
						use_batch_norm=configs.use_batch_norm)
                
	def forward(self, x, context=None):
		x = x.to(self.device)
		self.maf = self.maf.to(self.device)
		return self.maf.forward(x, context)

	def inverse(self, x, context=None):
		x = x.to(self.device)
		self.maf = self.maf.to(self.device)
		return self.maf.inverse(x, context)


class MAFAffine(nn.Module):
	''' Wrapper class for the MAF with affine transforms architecture
	'''
	def __init__(self, configs: dataclass):
		super(MAFAffine, self).__init__()
		self.dim = configs.dim_input
		self.device = configs.DEVICE
		self.num_transforms = configs.num_transforms
		self.flow_net = _MAFAffine(configs)
		self.get_permutation(configs.permutation)
		self.transforms()
		self.flows = CompositeTransform(self.transforms).to(configs.DEVICE)
		self.base_distribution = StandardNormal(shape=[configs.dim_input]).to(configs.DEVICE)
		self.net = Flow(self.flows, self.base_distribution)

	def get_permutation(self, perm):
		k = list(range(self.dim))
		if 'cycle' in perm:
			N = int(perm.split('-')[0]) 
			assert N < self.dim 
			self.permutation = Permutation(torch.tensor(k[-N:] + k[:-N]))
		elif 'reverse' in perm:
			self.permutation = Permutation(torch.tensor(k[::-1]))

	def transforms(self):
		self.transforms = []
		for _ in range(self.num_transforms):
			flow = copy.deepcopy(self.flow_net)
			perm = copy.deepcopy(self.permutation)
			self.transforms.append(flow)
			self.transforms.append(perm)

	def log_prob(self, x):
		return self.net.log_prob(x)
	
	def sample(self, num_samples):
		return self.net.sample(num_samples)

class _MAFAffine(nn.Module):
	def __init__(self, configs):
		super(_MAFAffine, self).__init__() 
		self.maf = MaskedAffineAutoregressiveTransform(
						features=configs.dim_input,
						hidden_features=configs.dim_hidden,
						num_blocks=configs.num_blocks,
						use_residual_blocks=configs.use_residual_blocks,
						dropout_probability=configs.dropout,
						use_batch_norm=configs.use_batch_norm)
                
	def forward(self, x, context=None):
		return self.maf.forward(x, context)

	def inverse(self, x, context=None):
		return self.maf.inverse(x, context)
	

#...Coupling layers:


class CouplingsPiecewiseRQS(nn.Module):
	''' Wrapper class for the MAF with rational quadratic splines architecture
	'''
	def __init__(self, configs: dataclass):
		super(CouplingsPiecewiseRQS, self).__init__()
		self.dim = configs.dim_input
		self.device = configs.DEVICE
		self.num_transforms = configs.num_transforms
		self.flow_net  = _CouplingsPiecewiseRQS(configs)
		self.get_permutation(configs.permutation)
		self.transforms()
		self.flows = CompositeTransform(self.transforms).to(configs.DEVICE)
		self.base_distribution = StandardNormal(shape=[configs.dim_input]).to(configs.DEVICE)
		self.net = Flow(self.flows, self.base_distribution)

	def get_permutation(self, perm):
		k = list(range(self.dim))
		if 'cycle' in perm:
			N = int(perm.split('-')[0]) 
			assert N < self.dim 
			self.permutation = Permutation(torch.tensor(k[-N:] + k[:-N]))
		elif 'reverse' in perm:
			self.permutation = Permutation(torch.tensor(k[::-1]))

	def transforms(self):
		self.transforms = []
		for _ in range(self.num_transforms):
			flow = copy.deepcopy(self.flow_net)
			perm = copy.deepcopy(self.permutation)
			self.transforms.append(flow)
			self.transforms.append(perm)

	def log_prob(self, x):
		return self.net.log_prob(x)
	
	def sample(self, num_samples):
		return self.net.sample(num_samples)


class _CouplingsPiecewiseRQS(nn.Module):
	''' Wrapper class for the Coupling layers with rational quadratic splines architecture
	'''
	def __init__(self, configs):
		super(_CouplingsPiecewiseRQS, self).__init__() 

		mask = torch.ones(configs.dim_input)
		if configs.mask == 'checkerboard': 
			mask[::2]=-1
		elif configs.mask == 'mid-split': 
			mask[int(configs.dim_input/2):]=-1  # 2006.08545
	
		def resnet(in_features, out_features):
			return ResidualNet(in_features,
								out_features,
								hidden_features=configs.dim_hidden,
								num_blocks=configs.num_blocks,
								dropout_probability=configs.dropout,
								use_batch_norm=configs.use_batch_norm)

		self.coupl = PiecewiseRationalQuadraticCouplingTransform(
							mask=mask,
        					transform_net_create_fn=resnet,
        					num_bins=configs.num_bins,
        					tails=configs.tails,
        					tail_bound=configs.tail_bound)

	def forward(self, x, context=None):
		return self.coupl.forward(x, context)

	def inverse(self, x, context=None):
		return self.coupl.inverse(x, context)
		
