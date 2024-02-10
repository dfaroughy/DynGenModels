
from DynGenModels.configs.utils import Register_Experiment
from DynGenModels.datamodules.mnist.configs import MNIST_Configs
from DynGenModels.models.configs import UNet_Configs
from DynGenModels.dynamics.cnf.configs import FlowMatch_Configs, CondFlowMatch_Configs
from DynGenModels.pipelines.configs import NeuralODE_Sampler_Configs

#...Register MNIST experiments:

MNIST_UNet_FlowMatch = Register_Experiment(data = MNIST_Configs,
                                           model = UNet_Configs, 
                                           dynamics = FlowMatch_Configs, 
                                           pipeline = NeuralODE_Sampler_Configs)

MNIST_UNet_CondFlowMatch = Register_Experiment(data = MNIST_Configs,
                                               model = UNet_Configs, 
                                               dynamics = CondFlowMatch_Configs, 
                                               pipeline = NeuralODE_Sampler_Configs)