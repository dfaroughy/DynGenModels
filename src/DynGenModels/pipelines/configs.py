from dataclasses import dataclass

@dataclass
class NeuralODE_Sampler_Configs:
    solver : str = 'euler'
    num_sampling_steps : int = 100
    sensitivity : str = 'adjoint'
    atol : float = 1e-4
    rtol : float = 1e-4

@dataclass
class NormFlows_Sampler_Configs:
    num_gen_samples: int = 1000     