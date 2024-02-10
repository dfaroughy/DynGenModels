from DynGenModels.configs.utils import Register_Experiment

from DynGenModels.datamodules.jetnet.configs import JetNet_Configs
from DynGenModels.models.configs import DeepSets_Configs, EPiC_Configs
from DynGenModels.dynamics.cnf.configs import FlowMatch_Configs, CondFlowMatch_Configs
from DynGenModels.pipelines.configs import NeuralODE_Sampler_Configs


JetNet_DeepSets_FlowMatch = Register_Experiment(data = JetNet_Configs,
                                                model = DeepSets_Configs, 
                                                dynamics = FlowMatch_Configs, 
                                                pipeline = NeuralODE_Sampler_Configs)

JetNet_DeepSets_CondFlowMatch = Register_Experiment(data = JetNet_Configs,
                                                    model = DeepSets_Configs, 
                                                    dynamics = CondFlowMatch_Configs, 
                                                    pipeline = NeuralODE_Sampler_Configs)

JetNet_EPiC_CondFlowMatch = Register_Experiment(data = JetNet_Configs,
                                                model = EPiC_Configs, 
                                                dynamics = CondFlowMatch_Configs, 
                                                pipeline = NeuralODE_Sampler_Configs)