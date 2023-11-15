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
    preprocess : List[str] = field(default_factory = lambda : ['normalize', 'logit_transform', 'standardize'])
    cuts_sideband_low : Dict[str, List[float]] = field(default_factory = lambda: {'mjj': [2700, 3100], } )
    cuts_sideband_high : Dict[str, List[float]] = field(default_factory = lambda: {'mjj': [3900, 13000]} )
    num_dijets : int = 60000
    def __post_init__(self):
        self.dim_input = len(self.features)

@dataclass
class LHCOlympics_LowLevel_Configs:
    DATA : str = 'LHCOlympicsLowLevel'
    dataset : str = '../../data/LHCOlympics2020/events_anomalydetection_high_level_cathode.h5'
    features : List[str] = field(default_factory = lambda : ['px1', 'py1', 'pz1', 'm1', 'px2', 'py2', 'pz2', 'm2',])
    dim_input : int = 8
    preprocess : List[str] = field(default_factory = lambda : ['normalize', 'logit_transform', 'standardize'])
    cuts_sideband_low : Dict[str, List[float]] = field(default_factory = lambda: {'mjj': [2700, 3100], } )
    cuts_sideband_high : Dict[str, List[float]] = field(default_factory = lambda: {'mjj': [3900, 13000]} )
    num_dijets : int = 60000
    def __post_init__(self):
        self.dim_input = len(self.features)

@dataclass
class LHCOlympics_Configs:
    DATA : str = 'LHCOlympics'
    dataset : str = '../../data/LHCOlympics2020/events_anomalydetection_dijets.h5'
    features : List[str] = field(default_factory = lambda : ['px_0', 'py_0', 'pz_0', 'm_0', 'px_1', 'py_1', 'pz_1', 'm_1'])
    dim_input : int = 8
    preprocess : List[str] = field(default_factory = lambda : ['log_pt', 'log_mass'])
    cuts_sideband_low : Dict[str, List[float]] = field(default_factory = lambda: {'mjj': [2700, 3100], } )
    cuts_sideband_high : Dict[str, List[float]] = field(default_factory = lambda: {'mjj': [3900, 13000]} )
    num_dijets : int = 60000
    def __post_init__(self):
        self.dim_input = len(self.features)