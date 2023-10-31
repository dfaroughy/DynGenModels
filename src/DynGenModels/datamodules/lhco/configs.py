import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict

""" Default configurations for LHC Olympics 2020 datasets.
"""

@dataclass
class LHCOlympics_Configs:
    DATA : str = 'LHCOlympics'
    dataset : str = '../../data/LHCOlympics2020/events_anomalydetection_dijets.h5'
    features : List[str] = field(default_factory = lambda : ['pt_1', 'eta_1', 'phi_1', 'm_1', 'pt_2', 'eta_2', 'phi_2', 'm_2'])
    dim_input : int = 8
    preprocess : List[str] = field(default_factory = lambda : ['log_pt', 'log_mass'])
    cuts_sideband_low : Dict[str, List[float]] = field(default_factory = lambda: {'mjj': [2700, 3100], } )
    cuts_sideband_high : Dict[str, List[float]] = field(default_factory = lambda: {'mjj': [3900, 13000]} )
    num_dijets : int = 60000
