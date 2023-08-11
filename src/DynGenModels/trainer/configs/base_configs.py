
from dataclasses import dataclass, field
from typing import List, Dict

@dataclass
class DataConfig:

    data_name : str = 'tops'
    features   : List[str] = field(default_factory = lambda : ['eta_rel', 'phi_rel'])
    preprocess : List[str] = field(default_factory = lambda : ['standardize'])
    datasets   : Dict[str, List[str]] = field(default_factory = lambda: {'name': ['file.hdf5', 'key']})
    labels : Dict[str, int] = field(default_factory = lambda:  {'name': 0} )
    num_jets : int = 1000000 
    num_constituents : int = 150

    def __post_init__(self):
        coords = ['eta_rel', 'phi_rel']
        self.features = [feature for feature in coords if feature not in self.features] + self.features

    
@dataclass
class TrainConfig:

    device : str = 'cpu'
    data_split_fracs : List[float] = field(default_factory = lambda : [0.5, 0.2, 0.3])  # train / val / test 
    batch_size : int = 1024
    epochs : int = 1000  
    early_stopping : int = 30 
    warmup_epochs : int = 100    
    lr : float = 0.001
    seed : int = 12345

