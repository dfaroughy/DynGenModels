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
    DIM_INPUT : int = 784
    INPUT_SHAPE : Tuple[float] = field(default_factory = lambda : (1, 28, 28))
