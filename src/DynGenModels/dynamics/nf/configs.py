from dataclasses import dataclass

@dataclass
class NormFlow_Configs:
    DYNAMICS : str = 'NormFlow'
    num_transforms: int = 8

@dataclass
class Deconvolution_NormFlow_Configs:
    DYNAMICS : str = 'DeconvolutionNormFlow'
    num_transforms: int = 5
    num_mc_draws: int = 30
