import torch.nn as nn

from nflows.transforms.autoregressive import MaskedPiecewiseRationalQuadraticAutoregressiveTransform
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform

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
		return self.maf.forward(x, context)

	def inverse(self, x, context=None):
		return self.maf.inverse(x, context)