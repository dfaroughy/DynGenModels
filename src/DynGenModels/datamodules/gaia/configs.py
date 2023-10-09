import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict

@dataclass
class DataConfigs:
    data_name : str = 'Gaia'
    dataset : List[str] = field(default_factory = lambda : ['../../data/gaia/data.angle_340.smeared_00.npy',
                                                            '../../data/gaia/data.angle_340.smeared_00.cov.npy'])
    features : List[str] = field(default_factory = lambda : ['x', 'y', 'z', 'vx', 'vy', 'vz'])
    dim_input : int = 6


#...custom configs:

@dataclass
class Gaia_Configs(DataConfigs):
    r_sun : List[float] = field(default_factory = lambda :[8.122, 0.0, 0.0208])
    preprocess : List[str] = field(default_factory = lambda : ['unit_ball_transform', 'radial_blowup', 'standardize' ])
    cuts : Dict[str, List[float]] = field(default_factory = lambda: {'r': [0.0, 3.5]} )
    
