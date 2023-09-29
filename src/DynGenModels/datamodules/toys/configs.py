
from dataclasses import dataclass, field
from typing import List

@dataclass
class DataConfigs:
    features : List[str] = field(default_factory = lambda : ['x', 'y'])
    data_name : str = 'toys'
    num_points : int = 10000
    dim_input : int = 2

#...custom configs:

@dataclass
class Gauss_2_Moons_Configs(DataConfigs):
    data_name : str = 'gauss_2_moons'
    gauss_8_scale : float = 2
    gauss_8_var : float = 0.1
    moon_2_noise : float = 0.2

@dataclass
class Gauss_Deconv_Configs(DataConfigs):
    data_name : str = 'gaussian_deconvolution'
    noise_cov : List[float] = field(default_factory = lambda : [[0.1,0],[0,1]])

