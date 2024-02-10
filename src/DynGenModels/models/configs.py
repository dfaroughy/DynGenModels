from dataclasses import dataclass
from DynGenModels.trainer.configs import Training_Configs

""" Default configurations for models.
"""

@dataclass
class MLP_Configs(Training_Configs):
    MODEL : str = 'MLP'
    DIM_HIDDEN : int = 128   
    dim_time_emb : int = None
    num_layers : int = 3
    activation : str = 'ReLU'

@dataclass
class ResNet_Configs(Training_Configs):
    MODEL : str = 'ResNet'
    DIM_HIDDEN  : int = 128 
    NUM_BLOCKS : int = 3
    NUM_BLOCK_LAYERS : int = 2

@dataclass
class UNet_Configs(Training_Configs):
    MODEL : str = 'Unet'
    DIM_HIDDEN : int = 32 # divisible by 32
    NUM_RES_BLOCKS : int = 1

@dataclass
class DeepSets_Configs(Training_Configs):
    MODEL : str = 'DeepSets'
    DIM_HIDDEN : int = 128   
    NUM_LAYERS_1 : int = 3
    NUM_LAYERS_2 : int = 3
    POOLING : str = 'meansum'

@dataclass
class EPiC_Configs(Training_Configs):
    MODEL : str = 'EPiC'
    POOLING: str = 'mean_sum'
    DIM_HIDDEN  : int = 128
    DIM_GLOBAL : int = 10
    NUM_EPIC_LAYERS : int = 6

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