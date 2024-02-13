from DynGenModels.configs.utils import Load_Experiment_Config

# Register JetNet experiments configs:

from DynGenModels.datamodules.jetnet.configs import JetNet_Config
from DynGenModels.models.architectures.configs import DeepSets_Config, EPiC_Config
from DynGenModels.dynamics.cnf.configs import FlowMatch_Config, CondFlowMatch_Config
from DynGenModels.pipelines.configs import NeuralODE_Sampler_Config

Config_JetNet_DeepSets_FlowMatch = Load_Experiment_Config(data = JetNet_Config,
                                                          model = DeepSets_Config, 
                                                          dynamics = FlowMatch_Config, 
                                                          pipeline = NeuralODE_Sampler_Config)

Config_JetNet_DeepSets_CondFlowMatch = Load_Experiment_Config(data = JetNet_Config,
                                                              model = DeepSets_Config, 
                                                              dynamics = CondFlowMatch_Config, 
                                                              pipeline = NeuralODE_Sampler_Config)

Config_JetNet_EPiC_CondFlowMatch = Load_Experiment_Config(data = JetNet_Config,
                                                          model = EPiC_Config, 
                                                          dynamics = CondFlowMatch_Config, 
                                                          pipeline = NeuralODE_Sampler_Config)