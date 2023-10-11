import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict

@dataclass
class DataConfigs:
    data_dir : str = '../../data/jetnet'
    data_name : str = 'JetNet'
    features : List[str] = field(default_factory = lambda : ['eta_rel', 'phi_rel', 'pt_rel'])
    dim_input : int = 3


#...custom configs:

@dataclass
class JetNet_Configs(DataConfigs):
    num_particles : int = 30
    jet_types : List[str] = field(default_factory = lambda : ['g', 'q', 't', 'w', 'z'])
    cuts : Dict[str, int] = field(default_factory = lambda : {'num_constituents': None})
    preprocess : List[str] = field(default_factory = lambda : ['standardize'])
    
    def __post_init__(self):
        self.data_name += str(self.num_particles)  