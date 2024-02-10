import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple

""" Default configurations for toy datasets.
"""

@dataclass
class MNIST_Configs:
    DATA_SOURCE = None
    DATA_TARGET = 'mnist'
    dim_input = (28, 28)
    data_split_fracs = [0.9, 0.1, 0]
    batch_size = 128
