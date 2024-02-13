from dataclasses import dataclass

""" Default configurations for discrete normalizing flow dynamics.
"""

@dataclass
class NormFlow_Config:
    DYNAMICS : str = 'NormFlow'
    permutation : str = '1-cycle'
    num_transforms: int = 5
    