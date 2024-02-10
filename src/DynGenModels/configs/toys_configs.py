
from DynGenModels.configs.utils import Register_Experiment
from DynGenModels.datamodules.toys.configs import Gauss_2_Moons_Configs, Gauss_2_Gauss_Configs
from DynGenModels.models.configs import MLP_Configs
from DynGenModels.dynamics.cnf.configs import FlowMatch_Configs
from DynGenModels.pipelines.configs import  NeuralODE_Sampler_Configs

#...Gauss 2 Moons Model Configs:

Gauss_2_Moons_MLP_FlowMatch = Register_Experiment(data = Gauss_2_Moons_Configs,
                                                model = MLP_Configs, 
                                                dynamics = FlowMatch_Configs, 
                                                pipeline = NeuralODE_Sampler_Configs)

Gauss_2_Gauss_MLP_FlowMatch = Register_Experiment(data = Gauss_2_Gauss_Configs,
                                                model = MLP_Configs, 
                                                dynamics = FlowMatch_Configs, 
                                                pipeline = NeuralODE_Sampler_Configs)
