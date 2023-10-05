import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple

@dataclass
class DataConfigs:
    features : List[str] = field(default_factory = lambda : ['x', 'y'])
    data_name : str = 'Deconvolution'
    num_points : int = 10000
    dim_input : int = 2

#...custom configs:

@dataclass
class Deconvolution_Gauss_Configs(DataConfigs):
    data_name : str = 'gaussian_deconvolution'
    noise_cov : List[List[float]] = field(default_factory = lambda : [[0.1, 0.0],[0.0, 1.0]])
    preprocess : List[str] = field(default_factory = lambda : [])
    cuts : Dict[str, List[float]] = field(default_factory = lambda: {'x': [-np.inf, np.inf], 'y': [-np.inf, np.inf]} )
    
