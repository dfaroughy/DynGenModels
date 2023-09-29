from typing import List
from dataclasses import dataclass, field

@dataclass
class Training_Configs:
    device : str = 'cpu'
    data_split_fracs : List[float] = field(default_factory = lambda : [0.7, 0.3, 0.0])  # train / val / test 
    batch_size : int = 256
    epochs : int = 10  
    early_stopping : int = 10 
    warmup_epochs : int = 10 
    print_epochs : int = 20   
    lr : float = 0.001
    seed : int = 12345
