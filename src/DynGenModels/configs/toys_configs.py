
import numpy as np
import json
from typing import List, Dict
from datetime import datetime
from dataclasses import dataclass, field, asdict
from DynGenModels.utils.utils import make_dir, print_table

@dataclass
class _DataConfig:
    data_name : str = 'toys'
    features   : List[str] = field(default_factory = lambda : ['x', 'y'])
    num_samples : int = 10000
    gauss_8_scale : float = 2
    gauss_8_var : float = 0.1
    moon_2_noise : float = 0.2
    noise_cov : List[float] = field(default_factory = lambda : [[0.1,0],[0,1]])

@dataclass
class _TrainConfig:
    device : str = 'cpu'
    data_split_fracs : List[float] = field(default_factory = lambda : [0.7, 0.3, 0.0])  # train / val / test 
    batch_size : int = 1024
    epochs : int = 1000  
    early_stopping : int = 30 
    warmup_epochs : int = 100 
    print_epochs : int = 10   
    lr : float = 0.001
    seed : int = 12345

@dataclass
class _SamplingConfig:
    solver : str = 'euler'
    num_sampling_steps : int = 100
    sensitivity : str = 'adjoint'
    atol : float = 1e-4
    rtol : float = 1e-4

@dataclass
class _DynamicsConfig:
    sigma : float = 0.1

#...Neural Network configarations:

@dataclass
class ToysMLPConfig(_DynamicsConfig, _SamplingConfig, _TrainConfig, _DataConfig):

    model_name : str = 'MLP'
    dim_input  : int = 3 
    dim_hidden : int = 128   

    def __post_init__(self):
        self.dim_input = len(self.features)

    def set_workdir(self, path: str='.', dir_name: str=None, save_config: bool=True):
        time = datetime.now().strftime("%Y.%m.%d_%Hh%M")
        dir_name = '{}.{}_{}'.format(self.model_name, self.data_name, time) if dir_name is None else dir_name
        self.workdir = make_dir(path + '/' + dir_name, overwrite=False)
        if save_config: self.save()

    def save(self, path: str=None):
        config = asdict(self)
        print_table(config)
        path = self.workdir + '/config.json' if path is None else path
        with open(path, 'w') as f:
            json.dump(config, f, indent=4)

    @classmethod
    def load(cls, path: str):
        with open(path, 'r') as json_file: config = json.load(json_file)
        print_table(config)
        return cls(**config)



@dataclass
class ToysResNetConfig(_DynamicsConfig, _SamplingConfig, _TrainConfig, _DataConfig):

    model_name : str = 'ResNet'
    dim_input  : int = 3 
    dim_hidden : int = 128   
    num_layers : int = 3

    def __post_init__(self):
        self.dim_input = len(self.features)

    def set_workdir(self, path: str='.', dir_name: str=None, save_config: bool=True):
        time = datetime.now().strftime("%Y.%m.%d_%Hh%M")
        dir_name = '{}.{}_{}'.format(self.model_name, self.data_name, time) if dir_name is None else dir_name
        self.workdir = make_dir(path + '/' + dir_name, overwrite=False)
        if save_config: self.save()

    def save(self, path: str=None):
        config = asdict(self)
        print_table(config)
        path = self.workdir + '/config.json' if path is None else path
        with open(path, 'w') as f:
            json.dump(config, f, indent=4)

    @classmethod
    def load(cls, path: str):
        with open(path, 'r') as json_file: config = json.load(json_file)
        print_table(config)
        return cls(**config)
