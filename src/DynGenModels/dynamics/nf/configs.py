from dataclasses import dataclass

""" Default configurations for discrete normalizing flow dynamics.
"""

@dataclass
class NormFlow_Configs:
    DYNAMICS : str = 'NormFlow'
    permutation : str = '1-cycle'
    num_transforms: int = 5
    
@dataclass
class Deconvolution_NormFlow_Configs:
    DYNAMICS : str = 'DeconvolutionNormFlow'
    permutation : str = '1-cycle'
    num_transforms: int = 5
    num_mc_draws: int = 30
