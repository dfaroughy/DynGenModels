from dataclasses import dataclass

""" Default configurations for sampling pipelines.
"""

@dataclass
class NeuralODE_Sampler_Config:
    SAMPLER : str = 'NeuralODE'
    SOLVER : str = 'euler'
    NUM_SAMPLING_STEPS : int = 100
    SENSITIVITY : str = 'adjoint'
    ATOL : float = None
    RTOL : float = None
    NUM_GEN_SAMPLES: int = 10000 

@dataclass
class NormFlows_Sampler_Config:
    SAMPLER : str = 'nflows'
    NUM_GEN_SAMPLES: int = 1000     