
from DynGenModels.configs.utils import Load_Experiment_Config

#...Register MNIST experiments configs:

from DynGenModels.datamodules.mnist.configs import MNIST_Config
from DynGenModels.models.architectures.configs import UNet_Config, UNetLight_Config
from DynGenModels.dynamics.cnf.configs import FlowMatch_Config, CondFlowMatch_Config
from DynGenModels.pipelines.configs import NeuralODE_Sampler_Config

Config_MNIST_UNet_FlowMatch = Load_Experiment_Config(data = MNIST_Config,
                                                     model = UNet_Config, 
                                                     dynamics = FlowMatch_Config, 
                                                     pipeline = NeuralODE_Sampler_Config)

Config_MNIST_UNet_CondFlowMatch = Load_Experiment_Config(data = MNIST_Config,
                                                         model = UNet_Config, 
                                                         dynamics = CondFlowMatch_Config, 
                                                         pipeline = NeuralODE_Sampler_Config)

Config_MNIST_UNetLight_CondFlowMatch = Load_Experiment_Config(data = MNIST_Config,
                                                              model = UNetLight_Config, 
                                                              dynamics = CondFlowMatch_Config, 
                                                              pipeline = NeuralODE_Sampler_Config)