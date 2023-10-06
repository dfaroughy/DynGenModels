from dataclasses import dataclass
from DynGenModels.trainer.configs import Training_Configs

@dataclass
class MLP_Configs(Training_Configs):
    model_name : str = 'MLP'
    dim_hidden : int = 128   
    num_layers : int = 3

@dataclass
class ResNet_Configs(Training_Configs):
    model_name : str = 'ResNet'
    dim_hidden : int = 128 
    num_blocks : int = 3
    num_block_layers : int = 2

@dataclass
class MAF_Affine_Configs(Training_Configs):
    model_name : str = 'MAF_Affine'
    dim_hidden : int = 128 
    num_blocks : int = 2 
    use_residual_blocks: bool = False
    dropout : float = 0.0
    use_batch_norm : bool = False

@dataclass
class Couplings_Affine_Configs(Training_Configs):
    model_name : str = 'Couplings_Affine'
    dim_hidden : int = 128 
    num_blocks : int = 2 
    use_residual_blocks: bool = False
    dropout : float = 0.0
    use_batch_norm : bool = False

@dataclass
class MAF_RQS_Configs(Training_Configs):
    model_name : str = 'MAF_RQS'
    dim_hidden : int = 128 
    num_blocks : int = 2 
    use_residual_blocks : bool= False
    dropout : float = 0.0
    use_batch_norm : bool = False
    num_bins : int = 10
    tails : str = 'linear'
    tail_bound : int = 5

@dataclass
class Couplings_RQS_Configs(Training_Configs):
    model_name : str = 'Couplings_RQS'
    mask : str = 'checkerboard'
    dim_hidden : int = 128 
    num_blocks : int = 2 
    use_residual_blocks : bool= False
    dropout : float = 0.0
    use_batch_norm : bool = False
    num_bins : int = 10
    tails : str = 'linear'
    tail_bound : int = 5