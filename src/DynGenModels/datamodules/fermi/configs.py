import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict

@dataclass
class FermiGCE_Configs:
    DATA : str = 'FermiGCE'
    dataset : str = '../../data/fermi/fermi_data_galactic_coord.npy'
    features : List[str] = field(default_factory = lambda : ['theta', 'phi', 'energy'])
    dim_input : int = 3
    preprocess : List[str] = field(default_factory = lambda : ['normalize', 'logit_transform', 'standardize'])
    cuts : Dict[str, List[float]] = field(default_factory = lambda: {'theta': [-10.0, 10.0], 'phi': [4.0, 10.0], 'energy': [1000.0, 2000.0]} )
    
