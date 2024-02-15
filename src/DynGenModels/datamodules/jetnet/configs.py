import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict

""" Default configurations for JetNet datasets.
"""

@dataclass
class DataConfig:
    NAME : str = None
    DATASET: str = 'jetnet'
    DATA_DIR : str = '../../data/jetnet'
    FEATURES : List[str] = field(default_factory = lambda : ['eta_rel', 'phi_rel', 'pt_rel'])
    DIM_INPUT : int = 3

#...custom configs:

@dataclass
class JetNet_Config(DataConfig):
    NUM_PARTICLES : int = 30
    JET_TYPES : List[str] = field(default_factory = lambda : ['g', 'q', 't', 'w', 'z'])
    CUTS : Dict[str, int] = field(default_factory = lambda : {'num_constituents': None})
    PREPROCESS : List[str] = field(default_factory = lambda : ['standardize'])
    