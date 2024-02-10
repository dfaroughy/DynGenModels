from dataclasses import dataclass

""" Default configurations for sampling pipelines.
"""

@dataclass
class NeuralODE_Sampler_Configs:
    SAMPLER : str = 'NeuralODE'
    SOLVER : str = 'euler'
    NUM_SAMPLING_STEPS : int = 100
    SENSITIVITY : str = 'adjoint'
    ATOL : float = 1e-4
    RTOL : float = 1e-4
    NUM_GEN_SAMPLES: int = 10000 

@dataclass
class NormFlows_Sampler_Configs:
    SAMPLER : str = 'nflows'
    NUM_GEN_SAMPLES: int = 1000     