
from DynGenModels.configs.utils import Load_Experiment_Config

# Register Toy experiments configs:

from DynGenModels.datamodules.toys.configs import Gauss_2_Moons_Config, Gauss_2_Gauss_Config
from DynGenModels.models.architectures.configs import MLP_Config
from DynGenModels.dynamics.cnf.configs import FlowMatch_Config
from DynGenModels.pipelines.configs import  NeuralODE_Sampler_Config


Config_Gauss_2_Moons_MLP_FlowMatch = Load_Experiment_Config(data = Gauss_2_Moons_Config,
                                                            model = MLP_Config, 
                                                            dynamics = FlowMatch_Config, 
                                                            pipeline = NeuralODE_Sampler_Config)

Config_Gauss_2_Gauss_MLP_FlowMatch = Load_Experiment_Config(data = Gauss_2_Gauss_Config,
                                                            model = MLP_Config, 
                                                            dynamics = FlowMatch_Config, 
                                                            pipeline = NeuralODE_Sampler_Config)
