from typing import List
from dataclasses import dataclass, field

""" Default configurations for training.
"""

@dataclass
class Optimizer_Configs:
    DEVICE : str = 'cpu'
    optimizer: str = 'Adam'
    lr : float = 1e-4
    weight_decay : float = 0.0
    optimizer_betas : List[float] = field(default_factory = lambda : [0.9, 0.999])
    optimizer_eps : float = 1e-8
    optimizer_amsgrad : bool = False
    gradient_clip : float = None

@dataclass
class Scheduler_Configs:
    scheduler: str = None
    scheduler_T_max: int = None
    scheduler_eta_min: float = None
    scheduler_gamma: float = None
    scheduler_step_size: int = None

@dataclass
class Training_Configs(Scheduler_Configs, Optimizer_Configs):
    EPOCHS: int = 10
    batch_size : int = 256
    data_split_fracs : List[float] = field(default_factory = lambda : [1.0, 0.0, 0.0])  # train / val / test 
    num_workers : int = 0
    pin_memory: bool = False
    early_stopping : int = None
    min_epochs : int = None 
    print_epochs : int = None   
    fix_seed : int = None