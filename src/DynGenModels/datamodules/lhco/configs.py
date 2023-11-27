import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict

""" Default configurations for LHC Olympics 2020 datasets.
"""

@dataclass
class LHCOlympics_HighLevel_Configs:
    DATA : str = 'LHCOlympicsHighLevel'
    dataset : str = '../../data/LHCOlympics2020/events_anomalydetection_high_level_cathode.h5'
    features : List[str] = field(default_factory = lambda : ['mjj', 'mj1', 'delta_m', 'tau21_1', 'tau21_2'])
    dim_input : int = 5
    exchange_target_with_source: bool = False
    preprocess : List[str] = field(default_factory = lambda : ['normalize'])
    cuts_sideband_low : Dict[str, List[float]] = field(default_factory = lambda: {'mjj': [2700, 3100], } )
    cuts_sideband_high : Dict[str, List[float]] = field(default_factory = lambda: {'mjj': [3900, 13000]} )
    num_dijets : int = 60000
    def __post_init__(self):
        self.dim_input = len(self.features)

@dataclass
class LHCOlympics_LowLevel_Configs:
    DATA : str = 'LHCOlympicsLowLevel'
    dataset : str = '../../data/LHCOlympics2020/events_anomalydetection_low_level_4mom.h5'
    features : List[str] = field(default_factory = lambda : ['px_j1', 'py_j1', 'pz_j1', 'e_j1', 'px_j2', 'py_j2', 'pz_j2', 'e_j2'])
    dim_input : int = 8
    exchange_target_with_source: bool = False
    preprocess : List[str] = field(default_factory = lambda : ['normalize'])
    cuts_sideband_low : Dict[str, List[float]] = field(default_factory = lambda: {'mjj': [2700, 3100], } )
    cuts_sideband_high : Dict[str, List[float]] = field(default_factory = lambda: {'mjj': [3900, 13000]} )
    num_dijets : int = 60000
    def __post_init__(self):
        self.dim_input = len(self.features)
