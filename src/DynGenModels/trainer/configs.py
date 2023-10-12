from typing import List
from dataclasses import dataclass, field

@dataclass
class Optimizer_Configs:
    DEVICE : str = 'cpu'
    optimizer: str = 'Adam'
    lr : float = 1e-4
    weight_decay : float = 0.0
    betas : List[float] = field(default_factory = lambda : [0.9, 0.999])
    eps : float = 1e-8
    amsgrad : bool = False
    gradient_clip : float = None

@dataclass
class Training_Configs(Optimizer_Configs):
    EPOCHS: int = 10
    batch_size : int = 256
    data_split_fracs : List[float] = field(default_factory = lambda : [1.0, 0.0, 0.0])  # train / val / test 
    num_workers : int = 0
    pin_memory: bool = False
    early_stopping : int = None
    min_epochs : int = None 
    print_epochs : int = None   
    seed : int = 12345