
from DynGenModels.configs.utils import Configs

from DynGenModels.datamodules.lhco.configs import LHCOlympics_LowLevel_Configs, LHCOlympics_HighLevel_Configs
from DynGenModels.models.configs import MLP_Configs, ResNet_Configs
from DynGenModels.dynamics.cnf.configs import CondFlowMatch_Configs
from DynGenModels.pipelines.configs import NeuralODE_Sampler_Configs



LHCOlympics_HighLevel_MLP_CondFlowMatch = Configs(data = LHCOlympics_HighLevel_Configs,
                                                    model = MLP_Configs, 
                                                    dynamics = CondFlowMatch_Configs, 
                                                    pipeline = NeuralODE_Sampler_Configs)

LHCOlympics_LowLevel_MLP_CondFlowMatch = Configs(data = LHCOlympics_LowLevel_Configs,
                                                model = MLP_Configs, 
                                                dynamics = CondFlowMatch_Configs, 
                                                pipeline = NeuralODE_Sampler_Configs)


