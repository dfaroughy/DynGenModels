from dataclasses import dataclass
from DynGenModels.trainer.configs import Training_Configs

""" Default configurations for models.
"""

@dataclass
class MLP_Configs(Training_Configs):
    MODEL : str = 'MLP'
    dim_hidden : int = 128   
    dim_time_emb : int = None
    num_layers : int = 3
    activation : str = 'ReLU'

@dataclass
class ResNet_Configs(Training_Configs):
    MODEL : str = 'ResNet'
    dim_hidden : int = 128 
    num_blocks : int = 3
    num_block_layers : int = 2

@dataclass
class DeepSets_Configs(Training_Configs):
    MODEL : str = 'DeepSets'
    dim_hidden : int = 128   
    num_layers_1 : int = 3
    num_layers_2 : int = 3
    poolings : str = 'meansum'

@dataclass
class EPiC_Configs(Training_Configs):
    MODEL : str = 'EPiC'
    pooling : str = 'mean_sum'
    dim_hidden : int = 128
    dim_global : int = 10
    num_epic_layers : int = 6

#...Normalizing Flow Models:

@dataclass
class MAF_Affine_Configs(Training_Configs):
    MODEL : str = 'MAF_Affine'
    dim_hidden : int = 128 
    num_blocks : int = 2 
    use_residual_blocks: bool = False
    dropout : float = 0.0
    use_batch_norm : bool = False

@dataclass
class Couplings_Affine_Configs(Training_Configs):
    MODEL : str = 'Couplings_Affine'
    dim_hidden : int = 128 
    num_blocks : int = 2 
    use_residual_blocks: bool = False
    dropout : float = 0.0
    use_batch_norm : bool = False

@dataclass
class MAF_RQS_Configs(Training_Configs):
    MODEL : str = 'MAF_RQS'
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
    MODEL : str = 'Couplings_RQS'
    mask : str = 'checkerboard'
    dim_hidden : int = 128 
    num_blocks : int = 2 
    use_residual_blocks : bool= False
    dropout : float = 0.0
    use_batch_norm : bool = False
    num_bins : int = 10
    tails : str = 'linear'
    tail_bound : int = 5