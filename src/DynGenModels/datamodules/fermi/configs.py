import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict

""" Default configurations for Fermi-LAT datasets.
"""

@dataclass
class FermiGCE_Config:
    DATASET : str = 'fermi'
    DATADIR : str = '../../data/fermi/fermi_data_galactic_coord.npy'
    FEATURES : List[str] = field(default_factory = lambda : ['theta', 'phi', 'energy'])
    DIM_INPUT : int = 3
    PREPROCESS : List[str] = field(default_factory = lambda : ['normalize', 'logit_transform', 'standardize'])
    CUTS : Dict[str, List[float]] = field(default_factory = lambda: {'theta': [-10.0, 10.0], 'phi': [4.0, 10.0], 'energy': [1000.0, 2000.0]} )
    
