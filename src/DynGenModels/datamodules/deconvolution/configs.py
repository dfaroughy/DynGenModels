import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple


""" Default configurations for deconvolution datasets.
"""

@dataclass
class Deconvolution_Gauss1D_Configs:
    DATA : str = 'Gauss1D'
    features : List[str] = field(default_factory = lambda : ['x'])
    num_points : int = 10000
    dim_input : int = 1
    log_norm_scale : float = 1.0
    preprocess : List[str] = field(default_factory = lambda : [])

@dataclass
class Deconvolution_Gauss2D_Configs:
    DATA: str = 'Gauss2D'
    features : List[str] = field(default_factory = lambda : ['x', 'y'])
    num_points : int = 10000
    dim_input : int = 2
    noise_cov : List[List[float]] = field(default_factory = lambda : [[0.1, 0.0],[0.0, 1.0]])
    preprocess : List[str] = field(default_factory = lambda : [])
    cuts : Dict[str, List[float]] = field(default_factory = lambda: {'x': [-np.inf, np.inf], 'y': [-np.inf, np.inf]} )
    
