import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple

""" Default configurations for toy datasets.
"""

@dataclass
class Gauss_2_Moons_Config:
    NAME : str = 'Gauss_2_Moons'
    features : List[str] = field(default_factory = lambda : ['x', 'y'])
    num_points : int = 10000
    dim_input : int = 2
    exchange_source_with_target : bool = False
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
class Gauss_2_Gauss_Config:
    NAME : str = 'Gauss_2_Gauss'
    features : List[str] = field(default_factory = lambda : ['x', 'y'])
    dim_input : int = 2
    num_points : int = 10000
    scale : float = 0.5

@dataclass
class Smeared_Gauss_Config:
    NAME : str = 'Gaussians'
    features : List[str] = field(default_factory = lambda : ['x', 'y'])
    num_points : int = 10000
    dim_input : int = 2
    exchange_source_with_target : bool = False
    noise_cov : List[List[float]] = field(default_factory = lambda : [[0.1, 0.0],[0.0, 1.0]])
    preprocess : List[str] = field(default_factory = lambda : [])
    cuts : Dict[str, List[float]] = field(default_factory = lambda: {'x': [-np.inf, np.inf], 'y': [-np.inf, np.inf]} )
    