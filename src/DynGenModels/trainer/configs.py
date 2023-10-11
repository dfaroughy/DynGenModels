from typing import List
from dataclasses import dataclass, field

@dataclass
class Training_Configs:
    device : str = 'cpu'
    data_split_fracs : List[float] = field(default_factory = lambda : [1.0, 0.0, 0.0])  # train / val / test 
    batch_size : int = 256
    epochs : int = 10  
    lr : float = 0.001
    early_stopping : int = None
    warmup_epochs : int = None 
    print_epochs : int = None   
    seed : int = 12345
    gradient_clip : float = None

