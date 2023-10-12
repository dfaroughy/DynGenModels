import torch
import torch.nn as nn

from nflows.transforms.autoregressive import MaskedPiecewiseRationalQuadraticAutoregressiveTransform
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform
from nflows.transforms.coupling import PiecewiseRationalQuadraticCouplingTransform
from nflows.nn.nets.resnet import ResidualNet

#...MAFs:

class MAFAffine(nn.Module):
	''' Wrapper class for the MAF with rational affine transforms architecture
	'''
	def __init__(self, configs):
		super(MAFAffine, self).__init__() 
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


class MAFPiecewiseRQS(nn.Module):
	''' Wrapper class for the MAF with rational quadratic splines architecture
	'''
	def __init__(self, configs):
		super(MAFPiecewiseRQS, self).__init__() 
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
	

#...Coupling layers:

class CouplingsPiecewiseRQS(nn.Module):
	''' Wrapper class for the Coupling layers with rational quadratic splines architecture
	'''
	def __init__(self, configs):
		super(CouplingsPiecewiseRQS, self).__init__() 

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
