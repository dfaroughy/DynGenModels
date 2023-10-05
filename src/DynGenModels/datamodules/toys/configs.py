import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple

@dataclass
class DataConfigs:
    features : List[str] = field(default_factory = lambda : ['x', 'y'])
    data_name : str = 'Toys'
    num_points : int = 10000
    dim_input : int = 2

#...custom configs:

@dataclass
class Gauss_2_Moons_Configs(DataConfigs):
    data_name : str = 'Gauss2Moons'
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
class Smeared_Gauss_Configs(DataConfigs):
    data_name : str = '3Gauss'
    noise_cov : List[List[float]] = field(default_factory = lambda : [[0.1, 0.0],[0.0, 1.0]])
    preprocess : List[str] = field(default_factory = lambda : [])
    cuts : Dict[str, List[float]] = field(default_factory = lambda: {'x': [-np.inf, np.inf], 'y': [-np.inf, np.inf]} )
    
