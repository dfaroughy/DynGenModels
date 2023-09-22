import json
from datetime import datetime
from dataclasses import dataclass, asdict
from DynGenModels.utils.utils import make_dir, print_table
from DynGenModels.configs.base_configs import TrainConfig, DataConfig

@dataclass
class ParticleNetConfig(TrainConfig, DataConfig):

    model_name  : str = 'ParticleNet'
    dim_input : int = 2
    dim_output : int = 2
    dim_hidden : int = 128 
    num_knn : int = 7     
    dim_conv_1 : int = 32
    dim_conv_2 : int = 64
    num_layers_1 : int = 3
    num_layers_2 : int = 3
    dropout : float = 0.1
    mkdir : bool = True

    def __post_init__(self):
        super().__post_init__()
        self.dim_input = len(self.features) - 2  # subtract (eta, phi)
        self.dim_output = len(self.datasets) - 1
        if self.mkdir:
            time = datetime.now().strftime("%Y.%m.%d_%H:%M:%S")
            self.workdir = make_dir('/home/df630/ClassifierMetric/results/{}.{}.{}_{}'.format(self.model_name, self.data_name, self.num_constituents, time), overwrite=True)

    def save(self, path):
        config = asdict(self)
        print_table(config)
        with open(path, 'w') as f:
            json.dump(config, f, indent=4)

    @classmethod
    def load(cls, path: str):
        with open(path, 'r') as json_file: config = json.load(json_file)
        print_table(config)
        config['mkdir'] = False
        return cls(**config)