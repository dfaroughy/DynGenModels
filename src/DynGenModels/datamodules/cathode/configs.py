import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict

""" Default configurations for LHC Olympics 2020 datasets.
"""

@dataclass
class Cathode_Configs:
    DATA : str = 'Cathode'
    data_gen_model : str = '../../data/LHCOlympics2020/events_anomalydetection_high_level_cathode.h5'
    data_reference : str = '../../data/LHCOlympics2020/events_anomalydetection_high_level_cathode_generated.h5'
    signal_noise_ratio : float = 0.01
    mass_window : List[float] = field(default_factory = lambda :[3300, 3700])
    features : List[str] = field(default_factory = lambda : ['mj1', 'delta_m', 'tau21_1', 'tau21_2'])
    dim_input : int = 4
    preprocess : List[str] = field(default_factory = lambda : ['stadardize'])
    def __post_init__(self):
        self.dim_input = len(self.features)
