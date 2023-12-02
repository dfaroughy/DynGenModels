import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict

""" Default configurations for LHC Olympics 2020 datasets.
"""

@dataclass
class Cathode_Classifier_Configs:
    DATA : str = 'Cathode'
    dataset_ref : str = '../../data/LHCOlympics2020/events_anomalydetection_high_level_cathode.h5'
    dataset_model : str = '../../data/LHCOlympics2020/events_anomalydetection_high_level_cathode_generated.h5'
    features : List[str] = field(default_factory = lambda : ['mj1', 'delta_m', 'tau21_1', 'tau21_2'])
    dim_input : int = 4
    exchange_target_with_source: bool = False
    preprocess : List[str] = field(default_factory = lambda : ['stadardize'])
    num_dijets : int = 60000
    def __post_init__(self):
        self.dim_input = len(self.features)
