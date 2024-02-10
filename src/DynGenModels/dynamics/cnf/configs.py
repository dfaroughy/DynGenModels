from dataclasses import dataclass

""" Default configurations for continious normalizing flow dynamics.
"""

@dataclass
class FlowMatch_Configs:
    DYNAMICS : str = 'FlowMatch'
    SIGMA : float = 1e-5
    T0 : float = 0.0
    T1 : float = 1.0

@dataclass
class CondFlowMatch_Configs:
    DYNAMICS : str = 'CondFlowMatch'
    SIGMA: float = 1e-5
    AUGMENTED : bool = False
    T0 : float = 0.0
    T1 : float = 1.0
