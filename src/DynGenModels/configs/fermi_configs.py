
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict

@dataclass
class DataConfig:
    dataset : str = None
    data_name : str = 'fermi_galactic_center'
    features   : List[str] = field(default_factory = lambda : ['theta', 'phi', 'energy'])
    preprocess : List[str] = field(default_factory = lambda : ['standardize'])
    cuts : Dict[str, int] = field(default_factory = lambda:  {'theta': [-np.inf, np.inf], 
                                                              'phi': [-np.inf, np.inf], 
                                                              'energy': [0.0, np.inf]} )
@dataclass
class TrainConfig:
    device : str = 'cpu'
    data_split_fracs : List[float] = field(default_factory = lambda : [0.7, 0.3, 0.0])  # train / val / test 
    batch_size : int = 1024
    epochs : int = 1000  
    early_stopping : int = 30 
    warmup_epochs : int = 100    
    lr : float = 0.001
    seed : int = 12345

@dataclass
class SamplingConfig:
    solver : str = 'euler'
    num_sampling_steps : int = 100
    sensitivity : str = 'adjoint'
    atol : float = 1e-4
    rtol : float = 1e-4

