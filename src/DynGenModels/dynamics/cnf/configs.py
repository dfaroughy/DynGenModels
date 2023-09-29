from dataclasses import dataclass

@dataclass
class CNF_Configs:
    objective : str = 'flow-matching'
    sigma : float = 0.1
    t0 : float = 0.0
    t1 : float = 1.0

@dataclass
class Deconvolution_CNF_Configs:
    objective : str = 'flow-matching'
    sigma : float = 1e-4
    t0 : float = 1.0
    t1 : float = 0.0