from dataclasses import dataclass

@dataclass
class FlowMatch_Configs:
    DYNAMICS : str = 'FlowMatch'
    sigma : float = 1e-5
    t0 : float = 0.0
    t1 : float = 1.0

@dataclass
class CondFlowMatch_Configs:
    DYNAMICS : str = 'CondFlowMatch'
    sigma : float = 0.1
    t0 : float = 0.0
    t1 : float = 1.0

@dataclass
class Deconvolution_CondFlowMatch_Configs:
    DYNAMICS : str = 'DeconvolutionMatch'
    sigma : float = 0.1
    t0 : float = 1.0
    t1 : float = 0.0