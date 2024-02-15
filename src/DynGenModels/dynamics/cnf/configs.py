from dataclasses import dataclass

""" Default configurations for continious normalizing flow dynamics.
"""

@dataclass
class FlowMatch_Config:
    DYNAMICS : str = 'FlowMatching'
    SIGMA : float = 1e-5
    T0 : float = 0.0
    T1 : float = 1.0

@dataclass
class CondFlowMatch_Config:
    DYNAMICS : str = 'ConditionalFlowMatching'
    SIGMA: float = 1e-5
    AUGMENTED : bool = False
    T0 : float = 0.0
    T1 : float = 1.0
