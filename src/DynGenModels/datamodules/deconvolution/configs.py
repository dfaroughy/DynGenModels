import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple

@dataclass
class Deconvolution_Gauss1D_Configs:
    data_name : str = 'Gauss1D'
    log_norm_scale : float = 1.0
    num_points : int = 10000
    dim_input : int = 1
    features : List[str] = field(default_factory = lambda : ['x'])
    preprocess : List[str] = field(default_factory = lambda : [])

@dataclass
class Deconvolution_Gauss2D_Configs:
    data_name : str = 'Gauss2D'
    num_points : int = 10000
    dim_input : int = 2
    features : List[str] = field(default_factory = lambda : ['x', 'y'])
    noise_cov : List[List[float]] = field(default_factory = lambda : [[0.1, 0.0],[0.0, 1.0]])
    preprocess : List[str] = field(default_factory = lambda : [])
    cuts : Dict[str, List[float]] = field(default_factory = lambda: {'x': [-np.inf, np.inf], 'y': [-np.inf, np.inf]} )
    
