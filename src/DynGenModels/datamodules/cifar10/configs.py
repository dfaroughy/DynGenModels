import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple

""" Default configurations for cifar datasets.
"""

@dataclass
class MNIST_Config:
    NAME : str = None
    DATASET : str = 'cifar'
    DATA_SOURCE : str = None
    DATA_TARGET : str = 'cifar10'
    DIM_INPUT : int = 3072
    INPUT_SHAPE : Tuple[float] = field(default_factory = lambda : (3, 32, 32))
