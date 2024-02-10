import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple

""" Default configurations for toy datasets.
"""

@dataclass
class MNIST_Configs:
    NAME : str = 'mnist'
    DATA_SOURCE : str = None
    DATA_TARGET : str = 'mnist'
    INPUT_SHAPE : Tuple[float] = field(default_factory = lambda : (1, 28, 28))
    DIM_INPUT : int = None

    def __post_init__(self):
        self.DIM_INPUT = np.prod(self.INPUT_SHAPE)
