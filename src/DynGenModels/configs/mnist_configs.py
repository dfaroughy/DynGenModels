
from DynGenModels.configs.utils import Configs
from DynGenModels.datamodules.mnist.configs import MNIST_Configs
from DynGenModels.models.configs import UNet_Configs
from DynGenModels.dynamics.cnf.configs import FlowMatch_Configs
from DynGenModels.pipelines.configs import NeuralODE_Sampler_Configs

#...Register MNIST experiments:

MNIST_UNet_FlowMatch = Configs(data = MNIST_Configs,
                               model = UNet_Configs, 
                               dynamics = FlowMatch_Configs, 
                               pipeline = NeuralODE_Sampler_Configs)