from DynGenModels.configs.utils import Configs

from DynGenModels.datamodules.gaia.configs import Gaia_Configs
from DynGenModels.models.configs import MLP_Configs, MAF_RQS_Configs, MAF_Affine_Configs
from DynGenModels.dynamics.cnf.configs import FlowMatch_Configs, CondFlowMatch_Configs
from DynGenModels.dynamics.nf.configs import NormFlow_Configs
from DynGenModels.pipelines.configs import NeuralODE_Sampler_Configs, NormFlows_Sampler_Configs


Gaia_MLP_FlowMatch = Configs(data = Gaia_Configs,
                            model = MLP_Configs, 
                            dynamics = FlowMatch_Configs, 
                            pipeline = NeuralODE_Sampler_Configs)
