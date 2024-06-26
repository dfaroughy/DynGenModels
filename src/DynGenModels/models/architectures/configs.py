from dataclasses import dataclass
from DynGenModels.models.configs import Training_Config

""" Default configurations for models.
"""

@dataclass
class MLP_Config(Training_Config):
    MODEL : str = 'MLP'
    DIM_HIDDEN : int = 128  
    TIME_EMBEDDING : str = 'sinusoidal'
    DIM_TIME_EMB : int = 16
    NUM_LAYERS : int = 3
    DROPOUT : float = 0.0
    ACTIVATION : str = 'ReLU'

@dataclass
class ResNet_Config(Training_Config):
    MODEL : str = 'ResNet'
    DIM_HIDDEN  : int = 128 
    NUM_BLOCKS : int = 3
    NUM_BLOCK_LAYERS : int = 2

@dataclass
class UnetCFM_Config(Training_Config):
    MODEL : str = 'Unet'
    DIM_HIDDEN : int = 32 # divisible by 32
    NUM_RES_BLOCKS : int = 1

@dataclass
class UnetNaive_Config(Training_Config):
    MODEL : str = 'UnetNaive'
    DIM_HIDDEN : int = 64 # divisible by 8
    DIM_TIME_EMB : int = 64

@dataclass
class Unet28x28_Config(Training_Config):
    MODEL : str = 'Unet28x28'
    DIM_HIDDEN : int = 64 
    DIM_TIME_EMB : int = 32
    ACTIVATION : str = 'GELU'
    DROPOUT : float = 0.1

@dataclass
class DeepSets_Config(Training_Config):
    MODEL : str = 'DeepSets'
    POOLING : str = 'mean_sum'
    DIM_HIDDEN : int = 64  
    TIME_EMBEDDING : str = 'sinusoidal'
    DIM_TIME_EMB : int = 4 
    NUM_LAYERS_PHI : int = 2
    NUM_LAYERS_RHO : int = 2
    DROPOUT : float = 0.0
    ACTIVATION : str = 'SELU'

@dataclass
class EPiC_Config(Training_Config):
    MODEL : str = 'EPiC'
    POOLING: str = 'mean_sum'
    DIM_HIDDEN  : int = 128
    TIME_EMBEDDING : str = 'sinusoidal'
    DIM_TIME_EMB : int = 16
    DIM_GLOBAL : int = 10
    NUM_EPIC_LAYERS : int = 6
    USE_SKIP_CONNECTIONS : bool = False
    ACTIVATION : str = 'ReLU'
    DROPOUT : float = 0.1
    
#...Normalizing Flow Models:

@dataclass
class MAF_Affine_Config(Training_Config):
    MODEL : str = 'MAF_Affine'
    dim_hidden : int = 128 
    num_blocks : int = 2 
    use_residual_blocks: bool = False
    dropout : float = 0.0
    use_batch_norm : bool = False

@dataclass
class Couplings_Affine_Config(Training_Config):
    MODEL : str = 'Couplings_Affine'
    dim_hidden : int = 128 
    num_blocks : int = 2 
    use_residual_blocks: bool = False
    dropout : float = 0.0
    use_batch_norm : bool = False

@dataclass
class MAF_RQS_Config(Training_Config):
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
class Couplings_RQS_Config(Training_Config):
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