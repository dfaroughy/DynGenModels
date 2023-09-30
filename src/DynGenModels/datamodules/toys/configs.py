import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple

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
    moon_2_noise : float = 0.2
    num_gaussians : int = 8
    gauss_N_scale : float = 5.0
    gauss_N_var : float = 0.1
    gauss_centers: List[Tuple[float, float]] = field(default_factory = lambda :[(1., 0.),
                                                                        (-1., 0.),
                                                                        (0., 1.),
                                                                        (0., -1.),
                                                                        (1.0 / np.sqrt(2), 1.0 / np.sqrt(2)),
                                                                        (1.0 / np.sqrt(2), -1.0 / np.sqrt(2)),
                                                                        (-1.0 / np.sqrt(2), 1.0 / np.sqrt(2)),
                                                                        (-1.0 / np.sqrt(2), -1.0 / np.sqrt(2))])

@dataclass
class Gauss_Deconv_Configs(DataConfigs):
    data_name : str = 'gaussian_deconvolution'
    noise_cov : List[List[float]] = field(default_factory = lambda : [[0.1, 0.0],[0.0, 1.0]])

