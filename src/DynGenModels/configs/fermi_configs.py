
import numpy as np
import json
from dataclasses import dataclass, field, asdict
from typing import List, Dict
from datetime import datetime

from DynGenModels.utils.utils import make_dir, print_table

@dataclass
class _DataConfig:
    dataset : str = None
    data_name : str = 'fermi_galactic_center'
    features   : List[str] = field(default_factory = lambda : ['theta', 'phi', 'energy'])
    preprocess : List[str] = field(default_factory = lambda : ['standardize'])
    cuts : Dict[str, int] = field(default_factory = lambda:  {'theta': [-np.inf, np.inf], 
                                                              'phi': [-np.inf, np.inf], 
                                                              'energy': [0.0, np.inf]} )
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
class FermiResNetConfig(_DynamicsConfig, _SamplingConfig, _TrainConfig, _DataConfig):

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


@dataclass
class FermiMLPConfig(_DynamicsConfig, _SamplingConfig, _TrainConfig, _DataConfig):

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