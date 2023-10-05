from dataclasses import dataclass

@dataclass
class FlowMatch_Configs:
    dynamics_name : str = 'FlowMatch'
    sigma : float = 1e-5
    t0 : float = 0.0
    t1 : float = 1.0

@dataclass
class CondFlowMatch_Configs:
    dynamics_name: str = 'CondFlowMatch'
    sigma : float = 0.1
    t0 : float = 0.0
    t1 : float = 1.0

@dataclass
class Deconvolution_CondFlowMatch_Configs:
    dynamics_name : str = 'DeconvolutionMatch'
    sigma : float = 0.1
    t0 : float = 1.0
    t1 : float = 0.0